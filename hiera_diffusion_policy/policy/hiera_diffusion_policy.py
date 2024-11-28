from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from einops import rearrange, reduce
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import DDPMScheduler
from hiera_diffusion_policy.so3diffusion.diffusion import SO3Diffusion
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
from hiera_diffusion_policy.so3diffusion.util import quat_to_rmat
import open3d as o3d

from hiera_diffusion_policy.model.common.normalizer import Normalizer
from hiera_diffusion_policy.policy.base_pcd_policy import BasePcdPolicy
from hiera_diffusion_policy.model.diffusion.guider import Guider
from hiera_diffusion_policy.model.diffusion.actor import Actor
from hiera_diffusion_policy.model.diffusion.critic import Critic2net
from hiera_diffusion_policy.common.visual import Color, draw_pcl
import hiera_diffusion_policy.common.transformation as tf
from hiera_diffusion_policy.model.diffusion.ema_model_hdp import EMAModel
from hiera_diffusion_policy.common.visual import visual_subgoals_tilt_v44_1, visual_subgoals_tilt_v44_2, visual_subgoals_v44_1, visual_subgoals_v446


class HieraDiffusionPolicy(BasePcdPolicy):
    def __init__(self, 
            guider: Guider,
            actor: Actor,
            critic: Critic2net,
            ema: EMAModel,
            noise_scheduler_guider: DDPMScheduler,
            noise_scheduler_actor: DDPMScheduler,
            horizon=16,
            action_dim=7,
            subgoal_dim=8,
            pcd_dim=3,
            subgoal_dim_nocont=6,
            n_action_steps=8,
            observation_history_num=2,
            use_pcd=True,
            discount=0.99,
            eta=1,
            single_step_reverse_diffusion=False,
            next_action_mode='pred_global',
            Tr=1,
            fin_rad=0.008,
            is_tilt=False
            ):
        super().__init__()

        # build guider
        self.guider = guider
        self.guider.train()
        self.guider_target = copy.deepcopy(self.guider)
        self.guider_target.eval()
        # build actor
        self.actor = actor
        self.actor.train()
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        # build critic
        self.critic = critic
        self.critic.train()
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        # build ema
        self.ema_actor = ema
        self.ema_critic = copy.deepcopy(self.ema_actor)
        self.ema_guider = copy.deepcopy(self.ema_actor)
        self.ema_actor.set_model(self.actor_target)
        self.ema_critic.set_model(self.critic_target)
        self.ema_guider.set_model(self.guider_target)

        self.noise_scheduler_guider = noise_scheduler_guider
        self.noise_scheduler_actor = noise_scheduler_actor
        self.normalizer = Normalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.pcd_dim = pcd_dim
        self.subgoal_dim = subgoal_dim
        self.subgoal_dim_nocont = subgoal_dim_nocont
        self.n_action_steps = n_action_steps
        self.observation_history_num = observation_history_num
        self.discount = discount
        self.eta = eta
        self.next_action_mode = next_action_mode
        self.use_pcd = use_pcd
        self.single_step_reverse_diffusion = single_step_reverse_diffusion
        self.Tr = Tr
        if is_tilt:
            self.fin_rad = fin_rad/0.01
        else:
            self.fin_rad = fin_rad
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


    # ***************** inference  *****************
    
    def predict_next_Q(self, batch):
        nbatch = self.normalizer.normalize(batch, self.subgoal_dim_nocont)
        B = nbatch['next_state'].shape[0]

        # current
        next_pcd = None
        if self.use_pcd:
            next_pcd = nbatch['next_pcd'].transpose(1, 2).reshape(
                    (B, -1, self.pcd_dim*self.observation_history_num))  # (B, 1024, 3n)
        next_state = nbatch['next_state'].reshape((B, -1))
        next_subgoal = nbatch['next_subgoal'] # (B, 8)
        next_action = nbatch['next_action'][:, self.observation_history_num-1:
                                            self.observation_history_num-1+self.Tr] # (B, A)
        next_action = next_action.reshape((B, -1))
        with torch.no_grad():
            current_q1, current_q2 = self.critic_target(
                next_pcd, next_state, next_subgoal, next_action)
        return torch.min(current_q1, current_q2)
    
    def predict_subgoal(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        nbatch = self.normalizer.normalize(obs_dict, self.subgoal_dim_nocont)
        B = nbatch['state'].shape[0]

        # init subgoal
        sg = torch.randn(size=(B, self.subgoal_dim),
                         dtype=self.dtype, device=self.device)  # (B, n*8+n)
        state = nbatch['state'].reshape((B, -1))
        pcd = None
        if self.use_pcd:
            pcd = nbatch['pcd'].transpose(1, 2).reshape(
                    (B, -1, self.pcd_dim*self.observation_history_num))  # (B, 1024, 3n)
        
        # ** loop denoise **
        with torch.no_grad():
            for t in self.noise_scheduler_guider.timesteps:
                # predict subgoal noise
                pred = self.guider_target(
                    pcd, state, sg, t)
                # compute previous subgoal
                sg = self.noise_scheduler_guider.step(
                    pred, t, sg, generator=None).prev_sample

        # ** output **
        sg[:, :6] = self.normalizer.unnormalize(nposition=sg[:, :6])
        sg[:, 6:] = torch.round(sg[:, 6:])
        sg[:, :3] *= sg[:, 6:7]
        sg[:, 3:6] *= sg[:, 7:]
        return sg
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预测action"""
        nobs = self.normalizer.normalize(obs_dict, self.subgoal_dim_nocont)
        B = nobs['state'].shape[0]
        
        pcd = None
        if self.use_pcd:
            pcd = nobs['pcd'].transpose(1, 2).reshape(
                (B, -1, self.pcd_dim*self.observation_history_num))  # (B, 1024, 3n)
            if 'pcd_id' in nobs:
                pcd = torch.concat((pcd, nobs['pcd_id']), dim=-1)

        state = nobs['state'].reshape((B, -1))
        subgoal = nobs['subgoal'] if 'subgoal' in nobs else None   # use subgoal
        
        with torch.no_grad():
            action = self.conditional_sample_action(pcd, state, subgoal)
        action = self.normalizer.unnormalize(naction=action)

        # get action
        start = self.observation_history_num - 1
        end = start + self.n_action_steps   # 1 + 8
        action_run = action[:,start:end]   # (B, 1:9, A)

        result = {
            'action': action_run,
            'action_pred': action,
        }
        return result
    
    def predict_next_action(self, obs_dict: Dict[str, torch.Tensor]):
        """预测next_action"""
        assert 'next_state' in obs_dict
        nobs = self.normalizer.normalize(obs_dict, self.subgoal_dim_nocont)
        B = nobs['next_state'].shape[0]

        next_pcd = None
        if self.use_pcd:
            next_pcd = nobs['next_pcd'].transpose(1, 2).reshape(
                (B, -1, self.pcd_dim*self.observation_history_num))
        next_state = nobs['next_state'].reshape((B, -1))
        next_subgoal = nobs['next_subgoal'] if 'next_subgoal' in nobs else None
        action_init = nobs['next_action'] if "next_action" in nobs else None

        with torch.no_grad():
            next_action = self.conditional_sample_action(
                next_pcd, next_state, next_subgoal, action_init)
        next_action = self.normalizer.unnormalize(naction=next_action)
        return next_action
    

    def conditional_sample_action(
            self, pcd, state, subgoal=None, action_init=None, model:Actor =None):
        """
        args:
            - pcd: (B, N, C)
            - state: (B, C)
            - subgoal: (B, C) if none, do not use
            - action_init: (B, T, C) if set, use as init
        
        return:
            - action (torch.Tensor): (B, C) normalized
        """
        # if action_init is not None:
        #     action = action_init
        #     timesteps = range(10)[::-1]
        # else:
        B = state.shape[0]
        shape = (B, self.horizon, self.action_dim)
        action = torch.randn(size=shape, dtype=self.dtype, device=self.device)
        timesteps = self.noise_scheduler_actor.timesteps

        for t in timesteps:
            if model is None:
                action_noise = self.actor_target(pcd, state, subgoal, action, t)
            else:
                action_noise = model(pcd, state, subgoal, action, t)
            # action
            action = self.noise_scheduler_actor.step(
                action_noise, t, action, generator=None).prev_sample
        return action
    

    # ***************** training  *****************
    def set_normalizer(self, normalizer: Normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss_guider(self, batch):
        # normalize input
        nbatch = self.normalizer.normalize(batch, self.subgoal_dim_nocont)
        subgoal = nbatch['subgoal']
        B = subgoal.shape[0]

        # ** add noise **
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler_guider.config.num_train_timesteps, # 100
            (B,), device=self.device
        ).long()

        # add noise
        noise = torch.randn(subgoal.shape, device=self.device)  # Sample noise
        noisy_sg = self.noise_scheduler_guider.add_noise(subgoal, noise, timesteps)

        # ** pred and loss **
        state = nbatch['state'].reshape((B, -1))
        pcd = None
        if self.use_pcd:
            pcd = nbatch['pcd'].transpose(1, 2).reshape(
                (B, -1, self.pcd_dim*self.observation_history_num))
        pred = self.guider(
            pcd, state, noisy_sg, timesteps)
        assert self.noise_scheduler_guider.config.prediction_type == 'epsilon'
        loss = F.mse_loss(pred, noise)
        return loss
    
    
    def compute_loss_critic(self, batch):
        # normalize input
        nbatch = self.normalizer.normalize(batch, self.subgoal_dim_nocont)
        B = nbatch['state'].shape[0]

        # current
        pcd = None
        if self.use_pcd:
            pcd = nbatch['pcd'].transpose(1, 2).reshape(
                    (B, -1, self.pcd_dim*self.observation_history_num))  # (B, 1024, 3n)
        state = nbatch['state'].reshape((B, -1))
        action = nbatch['action'][:, self.observation_history_num-1:
                                      self.observation_history_num-1+self.Tr]    # (B, A)
        # action = action.reshape((B, -1))
        subgoal = nbatch['subgoal'] # (B, 8)
        reward = nbatch['reward']   # (B, 1)
        dones = torch.zeros((B, 1), device=self.device)
        dones[reward==10] = 1

        # action随机加平移噪声，旋转噪声设为0
        # 噪声逻辑：先生成原始尺度噪声，再乘scale
        # 小噪声：最终action不加噪声, done不变
        # 大噪声: r=0, done=1
        if np.random.uniform() > 0.5:
            if np.random.uniform() > 0.5:
                # 小噪声
                noise = torch.randn(action.shape, device=self.device)*0.1
                scale = self.normalizer.params_dict['action']['scale']
                nscale = scale.expand_as(action)
                noise = torch.clip(noise, -self.fin_rad/2*nscale, self.fin_rad/2*nscale)
                
                # real_noise = (noise/nscale)*0.01
                # print('small noise:', real_noise[:5, :, :3])

                action[:, :-1] += noise[:, :-1]
            else:
                # 大噪声
                noise = torch.randn(action.shape, device=self.device)

                # scale = self.normalizer.params_dict['action']['scale']
                # nscale = scale.expand_as(action)
                # real_noise = (noise/nscale)*0.01
                # print('large noise:', real_noise[:5, :, :3])

                action += noise
                reward = torch.zeros((B, 1), device=self.device)
                dones = torch.ones((B, 1), device=self.device)
        

        """
        只添加平移噪声的方案: 效果不佳
        if np.random.uniform() > 0.5:
            if np.random.uniform() > 0.5:
                # 小噪声
                noise = torch.randn(action[:, :-1, :3].shape, device=self.device)*0.2  # 手指半径的倍数
                noise = torch.clip(noise, -0.5, 0.5)
                nscale = scale.expand_as(action[:, :-1, :3])
                action[:, :-1, :3] += noise*self.fin_rad*nscale
            else:
                # 大噪声
                noise = torch.randn(action[..., :3].shape, device=self.device)*20  # 手指半径的倍数
                nscale = scale.expand_as(action[..., :3])
                action[..., :3] += noise*self.fin_rad*nscale
                reward = torch.zeros((B, 1), device=self.device)
                dones = torch.ones((B, 1), device=self.device)
        """


        action = action.reshape((B, -1))
        current_q1, current_q2 = self.critic(
            pcd, state, subgoal, action)

        if self.next_action_mode == 'pred_local':
            # 使用actor预测next_action
            next_action_seq = self.predict_next_action(batch)
            batch['next_action'] = next_action_seq
        target_q = self.predict_next_Q(batch)

        target_q = (reward + (1-dones) * self.discount * target_q).detach()
        critic_loss = F.mse_loss(current_q1, target_q) + \
                      F.mse_loss(current_q2, target_q)
        return critic_loss


    def run_ema_critic(self):
        # ** critic ema **
        tau = 0.005
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def compute_loss_actor(self, batch):
        """ """
        nbatch = self.normalizer.normalize(batch, self.subgoal_dim_nocont)
        B = nbatch['state'].shape[0]

        # ******** bc loss ********
        # diffusion
        timesteps = torch.randint(
            0, self.noise_scheduler_actor.config.num_train_timesteps, # 100
            (B,), device=self.device
        ).long()
        # add noise to action
        noise = torch.randn(nbatch['action'].shape, device=self.device)  # Sample noise
        noisy_action = self.noise_scheduler_actor.add_noise(nbatch['action'], noise, timesteps)
        # pred
        pcd = None
        if self.use_pcd:
            pcd = nbatch['pcd'].transpose(1, 2).reshape(
                (B, -1, self.pcd_dim*self.observation_history_num))
            if 'pcd_id' in nbatch:
                pcd = torch.concat((pcd, nbatch['pcd_id']), dim=-1)
        state = nbatch['state'].reshape((B, -1))  # (B, n*S)
        subgoal = nbatch['subgoal'] if 'subgoal' in nbatch else None
        pred = self.actor(pcd, state, subgoal, noisy_action, timesteps)
        bc_loss = F.mse_loss(pred, noise)
        
        # ******** q loss ********
        if self.eta != 0:
            if self.single_step_reverse_diffusion:
                # 单次逆扩散，由Xt直接生成X0
                new_action_seq = self.noise_scheduler_actor.step_batch(
                        pred, timesteps, noisy_action, generator=None
                        ).pred_original_sample
            else:
                # 完整逆扩散
                new_action_seq = self.conditional_sample_action(
                    pcd, state, subgoal, model=self.actor)
            new_action = new_action_seq[:, self.observation_history_num-1:
                                        self.observation_history_num-1+self.Tr]
            new_action = new_action.reshape((B, -1))
            q1_new_action, q2_new_action = self.critic(
                pcd, state, subgoal, new_action)

            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()

            actor_loss = bc_loss + self.eta*q_loss

        else:
            q_loss = torch.tensor(-1, device=self.device)
            actor_loss = bc_loss
        
        return actor_loss, bc_loss, q_loss


    def test_guider(self, batch):
        """ """
        from hiera_diffusion_policy.common.pytorch_util import dict_apply

        Tbatch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
        subgoal = self.predict_subgoal(Tbatch).detach().to('cpu').numpy()
        B = batch['state'].shape[0]
        reward = np.zeros((B,))

        visual_pred_subgoals(
            state=batch['state'][:, -1],
            subgoal=subgoal,
            reward=reward,
            object_pcd=batch['pcd'][:, -1], 
            scene_pcd=batch['scene_pcd'][0])


def visual_pred_subgoals(state, subgoal, reward, scene_pcd, object_pcd):
    """可视化subgoal"""
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    for step in range(state.shape[0])[::20]:
        print('='*20)
        print('step:', step, 'reward:', reward[step])
        print('subgoal =', subgoal[step])
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        # visual scene
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # visual current object
        obj_color = np.array([[139, 105, 20]]).repeat(object_pcd[step].shape[0], axis=0)/255.
        ax.scatter(*tuple(object_pcd[step].transpose(1, 0)), color=obj_color)

        # visual current finger
        fl_pos = state[step, -6:-3]
        fr_pos = state[step, -3:]
        fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])
        fl_pcd_color = np.array([[0, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
        fr_pcd_color = np.array([[0, 0, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_pcd_color)
        ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_pcd_color)

        # visual subgoals
        # fl
        if subgoal[step, 6] == 1:
            fl_sg_pcd = tf.transPts_tq(finger_pcd, subgoal[step, :3], (0, 0, 0, 1))
            fl_sg_color = np.array([[255, 0, 0]]).repeat(fl_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fl_sg_pcd.transpose(1, 0)), color=fl_sg_color)
        # fr
        if subgoal[step, 7] == 1:
            fr_sg_pcd = tf.transPts_tq(finger_pcd, subgoal[step, 3:6], (0, 0, 0, 1))
            fr_sg_color = np.array([[34, 139, 34]]).repeat(fr_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fr_sg_pcd.transpose(1, 0)), color=fr_sg_color)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()

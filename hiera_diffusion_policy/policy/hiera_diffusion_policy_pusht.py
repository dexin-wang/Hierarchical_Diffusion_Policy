from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import DDPMScheduler
import copy
import numpy as np

from hiera_diffusion_policy.model.common.normalizer_pusht import LinearNormalizer
from hiera_diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
# from hiera_diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from hiera_diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from hiera_diffusion_policy.model.diffusion.guider import Guider
from hiera_diffusion_policy.model.diffusion.actor import Actor
from hiera_diffusion_policy.model.diffusion.critic import Critic2net
from hiera_diffusion_policy.model.diffusion.ema_model_hdp import EMAModel
from hiera_diffusion_policy.common.visual import visual_pushT_pred_subgoals


class HieraDiffusionPolicy(BaseLowdimPolicy):
    def __init__(self, 
            guider: Guider,
            actor: Actor,
            critic: Critic2net,
            ema: EMAModel,
            noise_scheduler_guider: DDPMScheduler,
            noise_scheduler_actor: DDPMScheduler,
            horizon,    # 16
            action_dim, # 10: 3+6+1
            obs_dim,
            subgoal_dim,
            subgoal_dim_nocont,
            n_action_steps,
            observation_history_num=2,
            discount=0.99,
            eta=1,
            single_step_reverse_diffusion=False,
            next_action_mode='pred_global',
            Tr=1,
            fin_rad=15,
            # parameters passed to step
            **kwargs):
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
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=obs_dim,
            max_n_obs_steps=observation_history_num,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.subgoal_dim = subgoal_dim
        self.subgoal_dim_nocont = subgoal_dim_nocont
        self.n_action_steps = n_action_steps
        self.observation_history_num = observation_history_num
        self.kwargs = kwargs
        self.discount = discount
        self.eta = eta
        self.next_action_mode = next_action_mode
        self.single_step_reverse_diffusion = single_step_reverse_diffusion
        self.Tr = Tr
        self.fin_rad = fin_rad
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
    # ========= inference  ============
    def predict_next_Q(self, batch):
        next_nobs = self.normalizer['obs'].normalize(batch['next_obs'])
        B = next_nobs.shape[0]
        next_nobs = next_nobs[:, :self.observation_history_num].reshape((B, -1))

        next_subgoal = self.normalizer['subgoal'].normalize(batch['next_subgoal'])
        next_action = self.normalizer['action'].normalize(batch['next_action'])
        next_action = next_action[:, self.observation_history_num-1: 
                                  self.observation_history_num-1+self.Tr] # (B, A)
        
        next_action = next_action.reshape((B, -1))

        with torch.no_grad():
            current_q1, current_q2 = self.critic_target(
                None, next_nobs, next_subgoal, next_action)
        return torch.min(current_q1, current_q2)
    

    def predict_subgoal(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测subgoal
        return: 
            sg: (B, 3) 手指子目标位置2/是否接触1
        """
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B = nobs.shape[0]
        nobs = nobs[:, :self.observation_history_num].reshape((B, -1))

        # init subgoal
        assert self.subgoal_dim == 3
        sg = torch.randn(size=(B, self.subgoal_dim),
                         dtype=self.dtype, device=self.device)  # (B, n*8+n)
        
        # ** loop denoise **
        with torch.no_grad():
            for t in self.noise_scheduler_guider.timesteps:
                # predict subgoal noise
                pred = self.guider_target(nobs, sg, t)
                # compute previous subgoal
                sg = self.noise_scheduler_guider.step(
                    pred, t, sg, generator=None).prev_sample

        # ** output **
        sg = self.normalizer['subgoal'].unnormalize(sg)
        sg[:, 2:] = torch.round(sg[:, 2:])
        sg[:, :2] *= sg[:, 2:3]
        return sg


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" and "subgoal" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict

        nobs = self.normalizer['obs'].normalize(obs_dict['obs']) 
        B, _, Do = nobs.shape

        subgoal = self.normalizer['subgoal'].normalize(obs_dict['subgoal']) if 'subgoal' in obs_dict else None
        shape = (B, self.horizon, self.action_dim+Do)
        cond_data = torch.zeros(size=shape, device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        cond_data[:, :self.observation_history_num, self.action_dim:] = nobs[:, :self.observation_history_num]  # 真值
        cond_mask[:, :self.observation_history_num, self.action_dim:] = True

        # run sampling
        with torch.no_grad():
            nsample = self.conditional_sample(cond_data, cond_mask, subgoal)
        
        # unnormalize prediction
        naction_pred = nsample[...,:self.action_dim] # 获取action
        action_pred = self.normalizer['action'].unnormalize(naction_pred)   # (batch, horizon, 2)

        # get action
        start = self.observation_history_num - 1
        end = start + self.n_action_steps   # 1 + 8
        action = action_pred[:,start:end]   # (batch, 1:9, 2) 预测16个action，只选用前8个
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result
    

    def predict_next_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" and "subgoal" key
        result: must include "action" key
        """
        assert 'next_obs' in obs_dict

        next_nobs = self.normalizer['obs'].normalize(obs_dict['next_obs'])
        B, _, Do = next_nobs.shape
        T = self.horizon    # 16
        Da = self.action_dim    # 2

        next_subgoal = self.normalizer['subgoal'].normalize(obs_dict['next_subgoal']) if 'next_subgoal' in obs_dict else None
        shape = (B, T, Da+Do)
        cond_data = torch.zeros(size=shape, device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        cond_data[:, :self.observation_history_num, Da:] = next_nobs[:, :self.observation_history_num]  # 真值
        cond_mask[:, :self.observation_history_num, Da:] = True

        data_init = None
        if 'next_action' in obs_dict:
            next_action = self.normalizer['action'].normalize(obs_dict['next_action'])
            data_init = torch.cat([next_action, next_nobs], dim=-1)   # (batch, hirizon, dim)

        # run sampling
        with torch.no_grad():
            nsample = self.conditional_sample(cond_data, cond_mask, next_subgoal, data_init)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da] # 获取action
        action_pred = self.normalizer['action'].unnormalize(naction_pred)   # (batch, horizon, 2)
        return action_pred
    

    def conditional_sample(self, 
            condition_data, condition_mask, 
            subgoal=None, data_init=None,
            model:Actor =None
            ):
        """
        condition_data: (batch, horizon, dim), 除GT的其他位置都是0
        condition_mask: (batch, horizon, dim), GT的位置为True
        """

        if data_init is not None:
            trajectory = data_init
            timesteps = range(10)[::-1]
        else:
            trajectory = torch.randn(
                size=condition_data.shape, dtype=self.dtype, device=self.device)
            timesteps = self.noise_scheduler_actor.timesteps

        for t in timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            if model is None:
                model_output = self.actor_target(trajectory, t, subgoal)
            else:
                model_output = model(trajectory, t, subgoal)

            # 3. compute previous image
            trajectory = self.noise_scheduler_actor.step(
                model_output, t, trajectory).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory
    

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def compute_loss_guider(self, batch):
        B = batch['obs'].shape[0]
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler_guider.config.num_train_timesteps, # 100
            (B,), device=self.device
        ).long()

        # add noise
        subgoal = self.normalizer['subgoal'].normalize(batch['subgoal'])
        noise = torch.randn(subgoal.shape, device=self.device)  # Sample noise
        noisy_sg = self.noise_scheduler_guider.add_noise(subgoal, noise, timesteps)

        # ** pred and loss **
        obs = self.normalizer['obs'].normalize(batch['obs'])
        obs = obs[:, :self.observation_history_num].reshape((B, -1))
        pred = self.guider(obs, noisy_sg, timesteps)
        assert self.noise_scheduler_guider.config.prediction_type == 'epsilon'
        loss = F.mse_loss(pred, noise)
        return loss


    def compute_loss_critic(self, batch):
        obs = self.normalizer['obs'].normalize(batch['obs'])
        B = obs.shape[0]
        obs = obs[:, :self.observation_history_num].reshape((B, -1))
        action = self.normalizer['action'].normalize(batch['action'])
        cur_action = action[:, self.observation_history_num-1: 
                            self.observation_history_num-1+self.Tr]    # (B, N, A)
        # cur_action = cur_action.reshape((B, -1))
        subgoal = self.normalizer['subgoal'].normalize(batch['subgoal'])
        reward = batch['reward']   # (B, 1)
        dones = torch.zeros((B, 1), device=self.device)
        dones[reward==10] = 1

        # action随机加噪声
        # 小噪声：最终action不加噪声, done不变
        # 大噪声: r=0, done=1
        if np.random.uniform() > 0.5:
            if np.random.uniform() > 0.5:
                # 小噪声
                noise = torch.randn(cur_action.shape, device=self.device)*0.1
                scale = self.normalizer.params_dict['action']['scale']
                nscale = scale.expand_as(cur_action)
                noise = torch.clip(noise, -self.fin_rad/2*nscale, self.fin_rad/2*nscale)
                cur_action[:, :-1] += noise[:, :-1]
            else:
                # 大噪声
                noise = torch.randn(cur_action.shape, device=self.device)*0.5

                # nscale = scale.expand_as(cur_action)
                # real_noise = noise/nscale
                # print('real noise:', real_noise[:5])
                # a

                cur_action += noise
                reward = torch.zeros((B, 1), device=self.device)
                dones = torch.ones((B, 1), device=self.device)

        cur_action = cur_action.reshape((B, -1))

        current_q1, current_q2 = self.critic(
            None, obs, subgoal, cur_action)

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

    def test_critic(self, batch):
        """
        测试critic
        """
        obs = self.normalizer['obs'].normalize(batch['obs'])
        B = obs.shape[0]
        obs = obs[:, :self.observation_history_num].reshape((B, -1))
        action = self.normalizer['action'].normalize(batch['action'])
        cur_action = action[:, self.observation_history_num-1: 
                            self.observation_history_num-1+self.Tr]    # (B, A)
        cur_action = cur_action.reshape((B, -1))
        subgoal = self.normalizer['subgoal'].normalize(batch['subgoal'])
        reward = batch['reward']   # (B, 1)
        dones = torch.zeros((B, 1), device=self.device)
        dones[reward==10] = 1

        current_q1, current_q2 = self.critic(
            None, obs, subgoal, cur_action)
        
        # print('cur_action =\n', cur_action[0])
        # print('q =', current_q1[0], current_q2[0])

        for i in range(B):
            # print('cur_action =\n', cur_action[i])
            print('r =', reward[i])
            print('q =', current_q1[i], current_q2[i])

        # # action随机加噪声, r设为0
        # for i in range(5):
        #     noise = torch.randn(cur_action.shape, device=cur_action.device)*0.1
        #     _cur_action = cur_action + noise

        #     current_q1, current_q2 = self.critic(
        #     None, obs, subgoal, _cur_action)

        #     print('=== 加噪 ===', i)
        #     print('noise =\n', noise[0]/0.004)
        #     print('cur_action =\n', _cur_action[0])
        #     print('q =', current_q1[0], current_q2[0])


    def compute_loss_actor(self, batch):
        """ """


        # normalize input
        assert 'valid_mask' not in batch
        assert 'obs' in batch
        assert 'action' in batch

        # ******** bc loss ********
        obs = self.normalizer['obs'].normalize(batch['obs'])
        action = self.normalizer['action'].normalize(batch['action'])
        B = obs.shape[0]

        # handle different ways of passing observation
        subgoal = self.normalizer['subgoal'].normalize(batch['subgoal']) if 'subgoal' in batch else None
        trajectory = torch.cat([action, obs], dim=-1)   # (batch, hirizon, dim)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        """
        condition_mask：
            shape: (batch, horizon, d)
            value: 同时满足 horizon < self.max_n_obs_steps 和 dim 为obs处为True，其他为False
            即[:, :self.max_n_obs_steps, action_dim:]
        """

        # *************** add noise ***************
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler_actor.config.num_train_timesteps, # 100
            (B,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler_actor.add_noise(
            trajectory, noise, timesteps)
        # compute loss mask
        loss_mask = ~condition_mask
        # apply conditioning
        # 将前2个timestep的obs设为GT
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # *************** Predict the noise residual ***************
        # pred的shape等于noisy_trajectory, (batch, horizon, dim)
        pred = self.actor(noisy_trajectory, timesteps, subgoal)
        pred_type = self.noise_scheduler_actor.config.prediction_type # 'epsilon'
        assert pred_type == 'epsilon'

        loss = F.mse_loss(pred, noise, reduction='none')   # 计算各元素的mse，不取均值, shape=(batch, horizon, dim)
        loss = loss * loss_mask.type(loss.dtype)    # 同时优化预测所有horizon的action和除掉前两个的obs
        loss = reduce(loss, 'b ... -> b (...)', 'mean') # 对于后面求均值来说，此行无作用？
        bc_loss = loss.mean()

        # ******** q loss ********
        if self.eta != 0:
            if self.single_step_reverse_diffusion:
                # 单次逆扩散，由Xt直接生成X0
                new_trajectory = self.noise_scheduler_actor.step_batch(
                        pred, timesteps, noisy_trajectory, generator=None
                        ).pred_original_sample
            else:
                # 完整逆扩散
                new_trajectory = self.conditional_sample(
                    trajectory, condition_mask, subgoal, model=self.actor)
            new_action_seq = new_trajectory[...,:self.action_dim]
            new_action = new_action_seq[:, self.observation_history_num-1:
                                        self.observation_history_num-1+self.Tr]
            new_action = new_action.reshape((B, -1))
            cur_obs = obs[:, :self.observation_history_num].reshape((B, -1))
            q1_new_action, q2_new_action = self.critic(
                None, cur_obs, subgoal, new_action)

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
        visual_pushT_pred_subgoals(batch['state'][:, self.observation_history_num-1], subgoal)



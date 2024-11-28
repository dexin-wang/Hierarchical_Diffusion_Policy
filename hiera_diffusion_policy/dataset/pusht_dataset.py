from typing import Dict
import torch
import numpy as np
import copy
from hiera_diffusion_policy.common.pytorch_util import dict_apply
from hiera_diffusion_policy.common.replay_buffer import ReplayBuffer
from hiera_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from hiera_diffusion_policy.model.common.normalizer_pusht import LinearNormalizer
from hiera_diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from hiera_diffusion_policy.common.robot import get_subgoals_pusht, get_subgoals_realtime_pusht
from hiera_diffusion_policy.common.visual import visual_pushT_dataset


"""
pusht任务使用的关键点为简化版点云，因此本任务不额外使用点云
"""

class PushTDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, # pusht_cchi_v7_replay.zarr 数据集路径
            horizon=1,      # 16
            observation_history_num=2,
            use_subgoal=True,
            pad_before=0,   # 1
            pad_after=0,    # 7
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            abs_action=True,
            seed=42,
            val_ratio=0.0,  # 0.02
            max_train_episodes=None,
            Tr=1
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])
        
        # print(self.replay_buffer['keypoint'].shape) # (25650, 9, 2)
        # print(self.replay_buffer['state'].shape)    # (25650, 5)
        # print(self.replay_buffer['action'].shape)   # (25650, 2)
        # print('episode_ends.shape =', self.replay_buffer.meta['episode_ends'])
        
        if use_subgoal:
            # 根据 meta['episode_ends'] 划分轨迹, 构建subgoal
            #! stage subgoal
            subgoals = get_subgoals_pusht(
                self.replay_buffer['state'],
                episode_ends=self.replay_buffer.meta['episode_ends'],
                fin_rad=15,
                sim_thresh=[3, 1./180*np.pi],
                max_reward=10,
                Tr=Tr,
                reward_mode='only_success'
                )
            #! realtime subgoal
            # subgoals = get_subgoals_realtime_pusht(
            #     self.replay_buffer['state'],
            #     episode_ends=self.replay_buffer.meta['episode_ends'],
            #     fin_rad=15,
            #     max_reward=10,
            #     Tr=Tr,
            #     reward_mode='only_success'
            #     )

            assert subgoals['subgoal'].shape[0] == self.replay_buffer['state'].shape[0]
            assert subgoals['next_subgoal'].shape[0] == self.replay_buffer['state'].shape[0]
            assert subgoals['reward'].shape[0] == self.replay_buffer['state'].shape[0]

            self.replay_buffer.data['subgoal'] =  subgoals['subgoal']
            self.replay_buffer.data['next_subgoal'] =  subgoals['next_subgoal']
            self.replay_buffer.data['reward'] =  subgoals['reward']

            #* 可视化数据集
            # visual_pushT_dataset(self.replay_buffer)

        # n_episodes = 200
        val_mask = get_val_mask(        # shape=(n_episodes,)  用于测试的episode为1，用于训练的为0
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        if max_train_episodes is not None:
            print(f'Use {max_train_episodes} demos to train!')
        else:
            print('Use all demos to train!')
        train_mask = downsample_mask(   # train_mask的shape不变，1的数量等于max_train_episodes
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            abs_action=abs_action,
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.observation_history_num = observation_history_num
        self.use_subgoal = use_subgoal
        self.abs_action = abs_action
        self.val_mask = val_mask
        self.Tr = Tr

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            abs_action=self.abs_action,
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask
        return val_set


    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample, next_sample=None, i=0):
        if isinstance(sample, ReplayBuffer):
            keypoint = sample[self.obs_key]
            agent_pos = sample[self.state_key][:,:2]
            obs = np.concatenate([
                keypoint.reshape(keypoint.shape[0], -1), 
                agent_pos], axis=-1)    # (20)
            # 用于计算归一化参数
            data = {
                'obs': obs, # (horizon, D_o)
                'action': sample[self.action_key], # (horizon, D_a)
            }
            if self.use_subgoal:
                subgoal_data = {
                    'subgoal': sample['subgoal'],   # (horizon, D_a)
                    }
                data.update(subgoal_data)

        else:
            keypoint = sample['data'][self.obs_key]
            agent_pos = sample['data'][self.state_key][:,:2]
            obs = np.concatenate([
                keypoint.reshape(keypoint.shape[0], -1), 
                agent_pos], axis=-1)    # (20)
            
            next_keypoint = next_sample['data'][self.obs_key]
            next_agent_pos = next_sample['data'][self.state_key][:,:2]
            next_obs = np.concatenate([
                next_keypoint.reshape(next_keypoint.shape[0], -1), 
                next_agent_pos], axis=-1)    # (20)
            # 用于训练和测试
            data = {
                'id': np.array([i,]),

                'state': sample['data'][self.state_key],    # only use to test

                'obs': obs, # (horizon, D_o)
                'action': sample['data'][self.action_key], # (horizon, D_a)

                'next_obs': next_obs, # (horizon, D_o)
                'next_action': next_sample['data'][self.action_key], # (horizon, D_a)
            }
            if self.use_subgoal:
                subgoal_data = {
                    'subgoal': sample['data']['subgoal'][self.observation_history_num-1],   # (3,)
                    'next_subgoal': sample['data']['next_subgoal'][self.observation_history_num-1], 
                    'reward': sample['data']['reward'][self.observation_history_num-1:self.observation_history_num], 
                    }
                data.update(subgoal_data)

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        return:
            - torch_data: dict() {
                'obs': shape=(horizon, D_o)
                'action': shape=(horizon, D_a)
            }
        """
        sample = self.sampler.sample_sequence(idx)
        next_idx = min(idx+self.Tr, len(self)-1)
        next_sample = self.sampler.sample_sequence(next_idx)
        data = self._sample_to_data(sample, next_sample, idx)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

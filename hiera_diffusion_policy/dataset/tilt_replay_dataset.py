from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import json
import copy
import hiera_diffusion_policy.common.transformation as tf
from hiera_diffusion_policy.common.robot import get_subgoals_stage_nonprehensile, get_subgoals_realtime_nonprehensile
from hiera_diffusion_policy.common.visual import visual_subgoals_v6
from hiera_diffusion_policy.common.pytorch_util import dict_apply
from hiera_diffusion_policy.dataset.base_dataset import BasePcdDataset
from hiera_diffusion_policy.model.common.normalizer import Normalizer
from hiera_diffusion_policy.model.common.rotation_transformer import RotationTransformer
from hiera_diffusion_policy.common.replay_buffer import ReplayBuffer
from hiera_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from hiera_diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)
import robomimic.utils.file_utils as FileUtils
import open3d as o3d

class TiltReplayDataset(BasePcdDataset):
    def __init__(self,
            dataset_path: str,
            observation_history_num=2,
            use_subgoal=True,
            horizon=1,  # 16
            pad_before=0,   # 1
            pad_after=0,    # 7
            obs_keys: List[str]=[
                'object_pos', 
                'object_quat', 
                'eef_pos', 
                'eef_quat',
                'fingers_position'],
            max_train_episodes=None,
            abs_action=False,   # True
            seed=42,
            Tr=1,
            val_ratio=0.02
        ):
        obs_keys = list(obs_keys)

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file['data']
            # 遍历轨迹，获取 state / action
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[f'demo_{i}']
                scene_pcd = demo['scene_pcd'][:].astype(np.float32)
                object_pcd = demo['object_pcd'][:].astype(np.float32)
                # get {state; action}
                data = _data_to_obs(
                    raw_obs=demo['obs'],
                    obj_pcd=object_pcd,
                    scene_pcd=scene_pcd,
                    raw_actions=demo['actions'][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    Tr=Tr)

                if use_subgoal:
                    # get subgoals and reward
                    #! stage subgoal
                    subgoals = get_subgoals_stage_nonprehensile(
                        demo['obs'],
                        object_pcd,
                        fin_rad=0.008,
                        sim_thresh=[0.02, 10./180*np.pi],
                        Tr=Tr,
                        reward_mode='only_success')
                    #! realtime subgoal
                    # subgoals = get_subgoals_realtime_nonprehensile(
                    #     demo['obs'],
                    #     object_pcd,
                    #     fin_rad=0.008,
                    #     Tr=Tr,
                    #     reward_mode='only_success')
                    data.update(subgoals)

                    #* 可视化每个状态的子目标
                    # visual_subgoals_v6(
                    #     state=data['state'],
                    #     subgoal=subgoals['subgoal'],
                    #     reward=subgoals['reward'],
                    #     object_pcd=object_pcd, 
                    #     scene_pcd=scene_pcd)

                replay_buffer.add_episode(
                    data, scene_pcd=scene_pcd, object_pcd=object_pcd)

        val_mask = get_val_mask(    # shape=(n_episodes,)  用于测试的episode为1，用于训练的为0
            n_episodes=replay_buffer.n_episodes,
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

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            abs_action=abs_action,
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.replay_buffer = replay_buffer
        self.use_subgoal = use_subgoal
        self.observation_history_num = observation_history_num
        self.sampler = sampler
        self.abs_action = abs_action
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
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


    def get_normalizer(self) -> Normalizer:
        normalizer = Normalizer()
        # action
        action_stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            action_params = robomimic_abs_action_only_normalizer_from_stat(action_stat)
        else:
            # already normalized
            action_params = get_identity_normalizer_from_stat(action_stat)
        normalizer.params_dict['action'] = action_params
        
        # state
        # offset为0，scale的所有元素相同
        state_stat = array_to_stats(self.replay_buffer['state'])
        normalizer.params_dict['state'] = normalizer_from_stat(state_stat)

        return normalizer
    
    def __len__(self):
        return len(self.sampler)

    def _sample_to_data(self, sample, i):
        data = {
            'id': np.array([i,]),

            'object_pcd': sample['meta']['object_pcd'],   # (1024, 3)
            'scene_pcd': sample['meta']['scene_pcd'],   # (1024, 3)

            'pcd': sample['data']['pcd'][:self.observation_history_num],   # (n, 1024, 3)
            'state': sample['data']['state'][:self.observation_history_num],  # (n, 27)
            'action': sample['data']['action'],

            'next_pcd': sample['data']['next_pcd'][:self.observation_history_num], # (n, 1024, 3)
            'next_state': sample['data']['next_state'][:self.observation_history_num],  # (n, 27)
            'next_action': sample['data']['next_action'],
        }
        if self.use_subgoal:
            subgoal_data = {
                'subgoal': sample['data']['subgoal'][self.observation_history_num-1],   # (8,)
                'next_subgoal': sample['data']['next_subgoal'][self.observation_history_num-1],   # (8,)
                'reward': sample['data']['reward'][self.observation_history_num-1:self.observation_history_num],
            }
            data.update(subgoal_data)

        return data


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return Normalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )


def _data_to_obs(raw_obs, obj_pcd, scene_pcd, raw_actions, obs_keys, abs_action, Tr):
    """
    args:
        raw_obs: h5py dict {}
        obj_pcd: 物体点云 (n, 3)
        scene_pcd: 场景点云 (n, 3)
        raw_actions: np.ndarray shape=(N, A) N为当前轨迹长度，A为action维度
        obs_keys: list(), 需要的观测, 是raw_obs.keys()的子集合

    return: Dict
        `state`: (N, S) S为需要的观测合并的维度： 机械臂末端位姿/两个手指的位置
        `action`: (N, A) 
        `next_state`: (N, A) 

    """
    assert abs_action == False
    
    obs = list()
    obs = np.concatenate([
        raw_obs[key] for key in obs_keys
    ], axis=-1).astype(np.float32)

    obj_pcd_batch = np.expand_dims(obj_pcd, axis=0).repeat(obs.shape[0], axis=0)
    obj_pcd_state = tf.transPts_tq_npbatch(obj_pcd_batch, obs[:, :3], obs[:, 3:7])  # (N, 1024, 3)
    # scene_pcd_batch = np.expand_dims(scene_pcd, axis=0).repeat(obs.shape[0], axis=0)    # (N, 1024, 3)
    # pcd_state = np.concatenate((obj_pcd_state, scene_pcd_batch), axis=1)    # (N, 2048, 3)

    data = {
        'pcd': obj_pcd_state[:-Tr],
        'state': obs[:-Tr],
        'action': raw_actions[:-Tr],

        'next_pcd': obj_pcd_state[Tr:],
        'next_state': obs[Tr:],
        'next_action': raw_actions[Tr:],
    }
    return data

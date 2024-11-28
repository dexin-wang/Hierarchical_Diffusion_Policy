from typing import Optional
import numpy as np
import gym
from gym.spaces import Box
import gym.spaces as spaces
from hiera_diffusion_policy.env.nonprehensile.rsuite import PushT3D


class PushT3DWrapper(gym.Env):
    def __init__(self, 
        env: PushT3D,
        init_state: Optional[np.ndarray]=None,
        render_hw=(256,256),
        render_camera_name='agentview'
        ):

        self.env = env
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = dict()
        self._seed = None
        self.pcd_n = 1024

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )

        self.reset()
        obs_example = self.get_observation()
        self.observation_space = spaces.Dict({
            'low_dim': spaces.Box(
                low=np.full_like(obs_example['low_dim'], fill_value=-1),
                high=np.full_like(obs_example['low_dim'], fill_value=1),
                shape=obs_example['low_dim'].shape,
                dtype=obs_example['low_dim'].dtype
            ),
            'object_pcd': spaces.Box(low=-2, high=2, shape=(self.pcd_n, 3), dtype=obs_example['object_pcd'].dtype)
        })

    def get_observation(self):
        """
        获取flatten的观测数据
        """
        raw_obs = self.env.get_observation()
        obs = self.updateState(raw_obs)
        return obs
    
    def updateState(self, raw_obs):
        """
        保留object_pose/eef_pos，在pusht3d任务中，eef_pos就是手指位置
        获取物体点云
        args:
            - raw_obs: dict, keys=['object', 'robot0_eef_pos', 'robot0_eef_quat']
        return:
            - obs: dict {
                - low_dim: np.ndarray shape=(7+3,) object_pose, eef_pos
                - object_pcd: (self.pcd_n, 3)
            }
        """
        object_pose = raw_obs['object-state'][:7]
        eef_pos = raw_obs['robot0_eef_pos']
        low_dim = np.concatenate((object_pose, eef_pos), axis=-1).astype(np.float32)
        object_pcd = self.env.get_object_pcd(self.pcd_n)
        return {'low_dim': low_dim, 'object_pcd': object_pcd}

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                np.random.seed(seed=seed)
                self.env.reset()
            else:
                # robosuite's initializes all use numpy global random state
                self.env.hard_reset = True
                np.random.seed(seed=seed)
                self.env.reset()
                self.env.hard_reset = False
                # state = self.env.get_state()['states']
                self.seed_state_map[seed] = None    #state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        # 空跑n step，等物体稳定
        action = np.zeros((self.action_space.shape[0]))
        for _ in range(10):
            self.env.step(action)
            
        # return obs
        obs = self.get_observation()
        return obs
    
    def reset_v1(self):
        self.env.reset()
        # return obs
        obs = self.get_observation()
        return obs
    
    def step(self, action):
        if action.shape[0] < self.action_space.shape[0]:
            _a = np.zeros((self.action_space.shape[0]))
            _a[:action.shape[0]] = action
        else:
            _a = action
        raw_obs, reward, done, info = self.env.step(_a)
        obs = self.updateState(raw_obs)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, 
            height=h, width=w, 
            camera_name=self.render_camera_name)

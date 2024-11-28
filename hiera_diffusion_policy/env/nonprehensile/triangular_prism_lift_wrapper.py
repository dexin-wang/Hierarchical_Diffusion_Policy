from typing import Optional
import numpy as np
import gym
from gym.spaces import Box
import gym.spaces as spaces
from hiera_diffusion_policy.common.visual import getFingersPos
from hiera_diffusion_policy.env.nonprehensile.rsuite import TriangularPrismLift


class TriangularPrismLiftWrapper(gym.Env):
    def __init__(self, 
        env: TriangularPrismLift,
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

    # def pcd_goal(self):
    #     return self.env.pcd_goal()
  
    def get_observation(self):
        """
        获取flatten的观测数据
        """
        raw_obs = self.env.get_observation()
        obs = self.updateState(raw_obs)
        return obs
    
    def updateState(self, raw_obs):
        """
        保留object_pose/eef_pose，计算 finger pos
        获取物体点云
        #! 此函数代码只适用于 PandaGripper, 如果更换机械手，需修改代码
        每个手指的中心点距离下边缘1cm，距离内侧边缘7.5mm
        args:
            - raw_obs: dict, keys=['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
        return:
            - obs: dict {
                - low_dim: np.ndarray shape=(7+7+6,) object_pose, eef_pos, eef_qua, fl_pos, fr_pos
                - object_pcd: (self.pcd_n, 3)
            }
        """
        object_pose = raw_obs['object-state'][:7]
        eef_pos = raw_obs['robot0_eef_pos']
        eef_qua = raw_obs['robot0_eef_quat']
        fl_pos, fr_pos = getFingersPos(
                raw_obs['robot0_eef_pos'], 
                raw_obs['robot0_eef_quat'], 
                raw_obs['robot0_gripper_qpos'][0]+0.0145/2, 
                raw_obs['robot0_gripper_qpos'][1]-0.0145/2)

        low_dim = np.concatenate((object_pose, eef_pos, eef_qua, fl_pos, fr_pos), axis=-1).astype(np.float32)
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
            
        # return obs
        obs = self.get_observation()
        return obs
    
    def reset_v1(self):
        self.env.reset()
        # return obs
        obs = self.get_observation()
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.updateState(raw_obs)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, 
            height=h, width=w, 
            camera_name=self.render_camera_name)

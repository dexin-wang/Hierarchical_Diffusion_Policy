from typing import List, Dict, Optional
import numpy as np
import gym
from gym.spaces import Box
import gym.spaces as spaces
from robomimic.envs.env_robosuite import EnvRobosuite
from hiera_diffusion_policy.common.visual import getFingersPos
from hiera_diffusion_policy.env.nonprehensile.rsuite import ManipulationGrasp
import hiera_diffusion_policy.env.nonprehensile.rsuite as suite

"""
参考本例程，Nonprehensile环境包含两个类：
NonprehensileEnv: 环境类，参考 robosuite 和 EnvRobosuite 重写
NonprehensilePcdWrapper: 封装类，参考 RobomimicPcdWrapper
"""


class TiltWrapper(gym.Env):
    def __init__(self, 
        env: ManipulationGrasp,
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
        self.pcd_n = 1024   #! 1024

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

        self.t_scale = 0.01 * 20 # 平移尺度
        self.r_scale = 5./180.*np.pi*2  # 旋转尺度
        self.f_scale = 0.003  # 每个机械手运动尺度，宽度是其2倍
        self.action_scale = np.array([self.t_scale, self.t_scale, self.t_scale, 
                                     self.r_scale, self.r_scale, self.r_scale, 
                                     self.f_scale])


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
        #! 此函数代码只适用于 SphereGripper, 如果更换机械手，需修改代码
        args:
            - raw_obs: dict, keys=['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
        return:
            - obs: dict {
                - low_dim: np.ndarray shape=(7+7+6,) object_pose, eef_pos, eef_qua, fl_pos, fr_pos
                - object_pcd: (self.pcd_n, 3)
            }
        """
        finger_half_size = 0.008
        object_pose = raw_obs['object-state'][:7]
        eef_pos = raw_obs['robot0_eef_pos']
        eef_qua = raw_obs['robot0_eef_quat']
        ld_f = raw_obs['robot0_gripper_qpos'][0] + 0.0185 + finger_half_size
        ld_r = raw_obs['robot0_gripper_qpos'][1] - 0.0185 - finger_half_size
        fl_pos, fr_pos = getFingersPos(eef_pos, eef_qua, ld_f, ld_r)

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

        # 空跑n step，等物体稳定
        action = np.zeros((self.action_space.shape[0]))
        for _ in range(5):
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
        raw_obs, reward, done, info = self.env.step(action*self.action_scale)
        obs = self.updateState(raw_obs)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, 
            height=h, width=w, 
            camera_name=self.render_camera_name)

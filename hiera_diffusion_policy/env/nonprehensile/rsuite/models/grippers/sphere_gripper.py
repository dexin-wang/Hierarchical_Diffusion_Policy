"""
Gripper with two fingers for Rethink Robots.
"""
import os
import numpy as np

from hiera_diffusion_policy.env.nonprehensile.rsuite.path import *
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class SphereGripperBase(GripperModel):
    """
    Gripper with long two-fingered parallel jaw.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(os.path.join(rs_assets_path(), "grippers/sphere_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.05, -0.05])
        # return np.array([0.020833, -0.020833])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["l_finger_g0", "l_finger_g1", "l_fingertip_g0", "l_fingerpad_g0"],
            "right_finger": ["r_finger_g0", "r_finger_g1", "r_fingertip_g0", "r_fingerpad_g0"],
            "left_fingerpad": ["l_fingerpad_g0"],
            "right_fingerpad": ["r_fingerpad_g0"],
        }


class SphereGripper(SphereGripperBase):
    """
    Modifies two finger base to only take one action.
    """

    def format_action(self, action):
        """
        将 机械手位移 映射为绝对位置
        self.current_action: 右手指/左手指 关节位置

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = self.current_action + np.array([1.0, -1.0]) * action
        # 关节限位
        self.current_action[0] = min(max(self.current_action[0], -0.012), 0.0185)
        self.current_action[1] = min(max(self.current_action[1], -0.0185), 0.012)

        return self.current_action


    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1

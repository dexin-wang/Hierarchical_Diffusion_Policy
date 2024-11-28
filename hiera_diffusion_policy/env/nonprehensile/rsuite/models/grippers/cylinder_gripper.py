"""
Gripper with two fingers for Rethink Robots.
"""
import os
from hiera_diffusion_policy.env.nonprehensile.rsuite.path import *
from robosuite.models.grippers.gripper_model import GripperModel



class CylinderGripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(os.path.join(rs_assets_path(), "grippers/cylinder_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action
    
    @property
    def init_qpos(self):
        return None

    # @property
    # def _important_geoms(self):
    #     return {
    #         "left_finger": ["l_finger_g0", "l_finger_g1", "l_fingertip_g0", "l_fingerpad_g0"],
    #         "right_finger": ["r_finger_g0", "r_finger_g1", "r_fingertip_g0", "r_fingerpad_g0"],
    #         "left_fingerpad": ["l_fingerpad_g0"],
    #         "right_fingerpad": ["r_fingerpad_g0"],
    #     }

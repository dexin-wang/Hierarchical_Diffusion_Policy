from .sphere_gripper import SphereGripper
from .cylinder_gripper import CylinderGripper
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite import ALL_GRIPPERS

GRIPPER_MAPPING["SphereGripper"] = SphereGripper
GRIPPER_MAPPING["CylinderGripper"] = CylinderGripper
ALL_GRIPPERS = GRIPPER_MAPPING.keys()

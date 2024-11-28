from collections import OrderedDict

import os
import numpy as np
from scipy.spatial.transform import Rotation as R

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from hiera_diffusion_policy.env.nonprehensile.rsuite.models.arenas import TableWallArenaY
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, BallObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
from hiera_diffusion_policy.env.nonprehensile.rsuite.path import *
import open3d as o3d
from robosuite.utils.mjcf_utils import postprocess_model_xml
from hiera_diffusion_policy.env.nonprehensile.rsuite.models.objects.xml_objects import TriangularPrismObject



class TriangularPrismLift(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        coord_x_range=[0, 0.1],
        coord_y_range=[-0.05, 0.05],
        rota_axis='z', # 'x','y','z'
        rota_range=[-0.2, 0.2],
        friction_range=[0.95, 0.95],
        density_range=[50, 50],
        table_friction_range=[0.95, 0.95],
        goal_hight=0.2,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.7, 0.7, 0.05),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=0, #!!! 默认-1
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        self.goal_hight = goal_hight
        # settings for table top
        self.table_full_size = table_full_size
        self.table_offset = np.array((0, 0, 0.8))

        self.coord_x_range = coord_x_range
        self.coord_y_range = coord_y_range
        self.rota_axis = rota_axis
        self.rota_range = rota_range
        self.friction_range = friction_range
        self.density_range = density_range
        self.table_friction_range = table_friction_range
        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1

        return reward

    # def pcd_goal(self):
    #     return None, None, self.object_goal_pose()
    
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        table_friction = (np.random.uniform(self.table_friction_range[0], self.table_friction_range[1]), 5e-3, 1e-4)
        # mujoco_arena = TableWallArenaY(
        #     table_full_size=self.table_full_size,
        #     table_friction=table_friction,
        #     table_offset=self.table_offset
        # )
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            # table_friction=table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.obj = TriangularPrismObject(name='triangular_prism')
        self.stl_path = self.obj.stl_path

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.obj)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.obj,
                x_range=[self.coord_x_range[0], self.coord_x_range[1]],
                y_range=[self.coord_y_range[0], self.coord_y_range[1]],
                rotation=(self.rota_range[0], self.rota_range[1]),
                rotation_axis=self.rota_axis, 
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.obj,
        )

    # **************************** diffusion_policy_pcd added ****************************
    def remove_all_objects(self):
        """
        move all objects to far places
        """
        object_placements = self.placement_initializer.sample()
        for obj_pos, obj_quat, obj in  object_placements.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array([5, 0, 5]), np.array([0, 0, 0, 1])]))
    
    def get_object_pcd(self, num=1024):
        """
        num: point count
        """
        mesh = o3d.io.read_triangle_mesh(self.stl_path)
        pcd = mesh.sample_points_uniformly(num, False)
        return np.asarray(pcd.points)

    def object_goal_pose(self):
        """
        目标位姿位于初始位姿上方4cm处，参考任务成功条件
        return: 7dim, [pos, quat(xyzw)]
        """
        return np.array([0, 0, 0, 0, 0, 0, 1])
    
    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.action_spec[0].shape[0]

    def get_observation(self):
        observations = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        return observations

    def render(self, mode="human", height=None, width=None, camera_name="agentview"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """

        if mode == "human":
            # cam_id = self.sim.model.camera_name2id(camera_name)
            # self.viewer.set_camera(cam_id)
            # return self.render()
            self.viewer.render()
        elif mode == "rgb_array":
            return self.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))
        
    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        xml = self.sim.model.get_xml() # model xml file
        state = np.array(self.sim.get_state().flatten()) # simulator state
        return dict(model=xml, states=state)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        should_ret = False
        if "model" in state:
            self.reset()
            xml = postprocess_model_xml(state["model"])
            self.reset_from_xml_string(xml)
            self.sim.reset()
            if not self._is_v1:
                # hide teleop visualization after restoring from model
                self.sim.model.site_rgba[self.eef_site_id] = np.array([0., 0., 0., 0.])
                self.sim.model.site_rgba[self.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in state:
            self.sim.set_state_from_flattened(state["states"])
            self.sim.forward()
            should_ret = True

        if "goal" in state:
            self.set_goal(**state["goal"])
        if should_ret:
            # only return obs if we've done a forward call - otherwise the observations will be garbage
            return self.get_observation()
        return None
    
    # **************************** diffusion_policy_pcd added done! ****************************

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = self.sim.model.body_name2id(self.obj.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # obj-related observables
            @sensor(modality=modality)
            def obj_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.obj_body_id])

            @sensor(modality=modality)
            def obj_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.obj_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_obj_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["obj_pos"]
                    if f"{pf}eef_pos" in obs_cache and "obj_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [obj_pos, obj_quat, gripper_to_obj_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the obj.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)
    
        # Color the gripper visualization site according to its distance to the obj
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.obj)
    

    def _check_success(self):
        """
        Check if obj has been lifted.

        Returns:
            bool: True if obj has been lifted
        """
        obj_height = self.sim.data.body_xpos[self.obj_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]
        return obj_height > table_height + self.goal_hight

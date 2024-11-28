import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from hiera_diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from hiera_diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from hiera_diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from hiera_diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from hiera_diffusion_policy.model.common.rotation_transformer import RotationTransformer

from hiera_diffusion_policy.policy.base_pcd_policy import BasePcdPolicy
from hiera_diffusion_policy.common.pytorch_util import dict_apply
from hiera_diffusion_policy.env_runner.base_pcd_runner import BasePcdRunner
# from hiera_diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from hiera_diffusion_policy.env.robomimic.robomimic_pcd_wrapper import RobomimicPcdWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from hiera_diffusion_policy.common.replay_buffer import ReplayBuffer
import hiera_diffusion_policy.common.transformation as tf
from hiera_diffusion_policy.common.visual import visual_subgoals_tilt_v44_1, visual_subgoals_tilt_v44_2, visual_pcd
import cv2


def create_env(env_meta, obs_keys, enable_render=True):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=enable_render, # 原始为False
        use_image_obs=False, 
    )
    return env


class RobomimicRunner(BasePcdRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            replay_buffer: ReplayBuffer,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            use_subgoal=True,
            use_pcd=True,
            observation_history_num=2,
            n_action_steps=8,
            n_latency_steps=0,
            # 渲染参数
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,   # true
            tqdm_interval_sec=5.0,
            n_envs=None,
            test_run=False
        ):
        """
        Assuming:
        observation_history_num=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = observation_history_num + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        rotation_transformer = None
        if abs_action:
            try:
                env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            except:
                env_meta['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicPcdWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys,
                    enable_render=False
                )
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicPcdWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                # init_state = f[f'data/demo_{train_idx}/states'][0]

                # def init_fn(env, init_state=init_state, 
                def init_fn(env, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicPcdWrapper)
                    # env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicPcdWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.use_subgoal = use_subgoal
        self.use_pcd = use_pcd
        self.observation_history_num = observation_history_num
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.replay_buffer = replay_buffer
        self.test_run = test_run


    def run(self, policy: BasePcdPolicy, first=False):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)  # 28
        n_inits = len(self.env_init_fn_dills)   # 56
        n_chunks = math.ceil(n_inits / n_envs)  # 向上取整 2

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            # 运行 init_fn(env), env_fn作为参数env
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            # past_action = None
            policy.reset()
            B = n_envs
            if first: return

            # **** 记录图像和轨迹 ****
            path = '/home/wdx/research/diffusion_robot_manipulation/trajectory_all_task/square'
            results_action = list()
            step = 0

            # 从replay_buffer获取 scene_pcd/object_pcd/goal
            scene_pcd = self.replay_buffer.scene_pcd    # (1024, 3)
            object_pcd = self.replay_buffer.object_pcd  # (1024, 3)
            scene_pcd = np.expand_dims(scene_pcd, axis=0).repeat(B, axis=0) # (B, 1024, 3)
            object_pcd = np.expand_dims(object_pcd, axis=0).repeat(B, axis=0) # (B, 1024, 3)

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {self.env_meta['env_name']}Pcd {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            # 在toolhang任务时，replay_buffer额外保存`tool_pcd`和`frame_frame`
            # obs的最后1位为0/1，0表示操作物体为frame, 1表示操作物体为tool;
            # 根据标志位将对应的物体点云与`replay_buffer.scene_pcd_ori`合并

            done = False
            nnn = 0
            while not done:
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    # 环境实际返回的观测包含5个观测，算法利用的观测只有前5-n_latency_steps个，模拟延迟获得n_latency_steps个观测数据
                    'state': obs[:,:self.observation_history_num].astype(np.float32)  # (28, 2, 7+7+6)
                }

                state = np_obs_dict['state']

                # *** 记录图像 ***
                # img = env.render_img()[10][..., ::-1]
                # cv2.imwrite(path+"/{:03d}.png".format(step), img)

                if self.use_pcd:
                    # 计算历史物体点云 (B, 2, 1024, 3)
                    obj_pcd = list()
                    for h in range(self.observation_history_num):
                        obj_pcd_h = tf.transPts_tq_npbatch(
                            object_pcd, state[:, h, :3], state[:, h, 3:7])
                        obj_pcd.append(obj_pcd_h)
                    obj_pcd = np.array(obj_pcd).transpose(1, 0, 2, 3)   # (n, B, 1024, 3)->(B, n, 1024, 3)
                    np_obs_dict['pcd'] = obj_pcd

                if self.use_subgoal:
                    # 预测子目标
                    Tinput_dict = dict_apply(np_obs_dict, 
                                            lambda x: torch.from_numpy(x).to(device=device))
                    subgoal = policy.predict_subgoal(Tinput_dict).detach().to('cpu').numpy()    # (B, 8)
                    np_obs_dict['subgoal'] = subgoal 

                    #! 可视化状态和子目标
                    nnn += 1
                    # if self.test_run:
                    #     b = 0
                    #     print('*'*10, 'b =', b, '*'*10)
                    #     print('subgoal =', np_obs_dict['subgoal'][b])
                    #     visual_subgoals_tilt_v44_2(
                    #         state[b, -1], np_obs_dict['subgoal'][b], scene_pcd[b], object_pcd[b])

                # device transfer
                Tinput_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(Tinput_dict)    # dict{'action', 'action_pred'}

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)

                # **** 记录action ****
                # 修改了multistep_wrapper的输出
                results_action.append(info[10])
                
                done = np.all(done)
                # past_action = action
                # update pbar
                pbar.update(action.shape[1])

                step += self.n_action_steps

            # **** 保存手指位置 ****
            # results_action = np.concatenate(tuple(results_action), axis=0)
            # np.save(path+'/action.npy', results_action)
            # print('轨迹记录完成!')
            # print('results_action.shape =', results_action.shape)

            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):  #!!!!!!!!!!!!
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
    

    def undo_transform_action(self, action):
        # raw_shape = action.shape
        # if raw_shape[-1] == 20:
        #     # dual arm
        #     action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4    # 6
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        # if raw_shape[-1] == 20:
        #     # dual arm
        #     uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

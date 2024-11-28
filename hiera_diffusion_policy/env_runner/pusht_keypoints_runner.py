import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import time
import math
import wandb.sdk.data_types.video as wv
from hiera_diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from hiera_diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from hiera_diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from hiera_diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from hiera_diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from hiera_diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from hiera_diffusion_policy.common.pytorch_util import dict_apply
from hiera_diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from hiera_diffusion_policy.common.visual import visual_pushT_pred_subgoal
import cv2
import copy
import os


# img = cv2.imread('/home/wdx/research/diffusion_robot_manipulation/hierachical_diffusion_policy_v6/pushT_init_2.png')
# manual_sgs = list()
# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(img, (x, y), 6, (255, 0, 0), thickness=-1)
#         cv2.imshow("image", img)
#         manual_sgs.append(np.array([x, y]))



class PushTKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            replay_buffer,
            keypoint_visible_rate=1.0,
            n_train=10,         # 6
            n_train_vis=3,      # 2
            train_start_seed=0,
            n_test=22,          # 50
            n_test_vis=6,       # 4
            legacy_test=False,  # True
            test_start_seed=10000,  # 100000
            max_steps=200,      # 300
            use_subgoal=False,
            observation_history_num=8,      # 2
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None     # null
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = observation_history_num + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
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
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
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

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
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

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.observation_history_num = observation_history_num
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.use_subgoal = use_subgoal
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseLowdimPolicy, first=False):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)  # 56
        n_inits = len(self.env_init_fn_dills)   # 56
        # pusht任务中，n_envs=n_inits
        n_chunks = math.ceil(n_inits / n_envs)  # ceil: 上整数

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs  # 0
            end = min(n_inits, start + n_envs)  # 56
            this_global_slice = slice(start, end)   # 截取 start->end
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)  # 截取 0->56
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()   # (56, 2, 46)
            past_action = None
            policy.reset()
            B = n_envs

            if first: return


            # 绘制图像
            # print('obs =', obs[0, self.observation_history_num-1, 18:23])
            # img = env.render_img()[0]
            # cv2.imshow('img', img)
            # cv2.waitKey()

            # cv2.namedWindow("image")
            # cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
            # cv2.imshow("image", img)

            # results_path = '/home/wdx/research/diffusion_robot_manipulation/hierachical_diffusion_policy_v6/paper_results/control/control_pushT_right_1Point_2.npy'
            # results_path = '/home/wdx/research/diffusion_robot_manipulation/trajectory_all_task/pushT'
            # results_action = list()
            # results_subgoal = list()

            #! 记录指定的子目标
            # while (True):
            #     try:
            #         k = cv2.waitKey(100)
            #         if k == 27:
            #             break
            #     except Exception:
            #         cv2.destroyAllWindows()
            #         break
            # cv2.destroyAllWindows()
            # manual_sgs按环境数量进行复制
            # manual_sgss = list()
            # for _ in range(B):
            #     manual_sgss.append(copy.deepcopy(manual_sgs))

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            step = 0
            while not done:
                Do = obs.shape[-1] // 2 - 3 # obs多了物体pose
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.observation_history_num,:Do].astype(np.float32),   # obs前一半为obs,后一半为obs_mask
                    'obs_mask': obs[...,:self.observation_history_num,Do:2*Do] > 0.5
                }

                # *** 记录图像 ***
                # img = env.render_img()[10]
                # cv2.imwrite(results_path+"/{:03d}.png".format(step), img)
                
                if self.use_subgoal:
                    # 预测子目标
                    # device transfer
                    obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                    with torch.no_grad():
                        subgoal = policy.predict_subgoal(obs_dict).detach().to('cpu').numpy()
                        np_obs_dict['subgoal'] = subgoal
                    
                    #! 测试用
                    # results_subgoal.append(subgoal)

                    #* 可视化子目标
                    # for b in range(B)[:5]:
                    # 失败:2 3
                    # b = 3
                    # print('*'*10, 'b =', b, '*'*10)
                    # print('subgoal =', subgoal[b])
                    # visual_pushT_pred_subgoal(
                    #     obs[b,self.observation_history_num-1,18:23].astype(np.float32), 
                    #     subgoal[b])
                    # try:
                    #     print('reward =', reward[b])
                    # except:
                    #     pass          

                #! 设置子目标
                # for b in range(B):
                #     if len(manual_sgss[b]) > 0:
                #         np_obs_dict['subgoal'][b, :2] = manual_sgss[b][0]

                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)   # action, action_pred, action_obs_pred, obs_pred

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                
                #! 记录action，绘图用
                # results_action.append(action[10])
                # 去除action路过的子目标
                # for b in range(B):
                #     del_end = -1
                #     for sgi in range(len(manual_sgss[b])):
                #         sg = manual_sgss[b][sgi]
                #         for ai in range(action[b].shape[0]):
                #             # 判断sg 与 action[0, ai] 的距离，小于手指半径为靠近
                #             dist = np.linalg.norm(sg-action[b, ai])
                #             if dist < 30:
                #                 del_end = sgi
                #                 break
                #         if del_end < sgi:
                #             break
                #     # 删除子目标
                #     if del_end >= 0:
                #         del manual_sgss[b][:del_end+1]

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])

                step += self.n_action_steps
            pbar.close()

            #! 保存实验数据
            # results_action = np.concatenate(tuple(results_action), axis=0)
            # print('results_action.shape =', results_action.shape)
            # np.save(results_path+'/action.npy', results_action)
            # save_path = results_path.replace('.npy', '_subgoal.npy')
            # np.save(save_path, manual_sgs)
            # 保存预测的子目标
            # results_subgoal = np.array(results_subgoal).transpose((1, 0, 2)) # (N, B, 2)
            # save_path = results_path.replace('.npy', '_subgoal_predicted.npy')
            # np.save(save_path, results_subgoal)
            # print('save done!')

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]   # (环境数量, max_steps)
            
        # import pdb; pdb.set_trace()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
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

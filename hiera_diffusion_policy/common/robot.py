import hiera_diffusion_policy.common.transformation as tf
import numpy as np
from hiera_diffusion_policy.common.visual import segmentation_to_rgb, getFingersPos, show_rgb_seg_dep, show_pcd
import robomimic.utils.file_utils as FileUtils
from scipy.spatial.transform import Rotation as R
import math
import shapely
import copy
import torch
from hiera_diffusion_policy.common.visual import create_pusht_pts


def get_subgoals_stage_moveT(
        state: dict,
        object_pcd: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        Tr=1,
        reward_mode='tanh'
        ):
    """
    计算手指位置子目标, 不考虑平滑

    args:
        - state: np.ndarray (20,) {eef_pos, eef_quat, fingers_position，object_pos, object_quat}
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - sim_thresh: list(平移误差, 弧度误差) 计算物体位姿子目标是否达到的阈值，各任务单独设置
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """

    """
    (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    (2) 过滤：当记录的两次相邻物体位姿相似时(阈值很小)，删除前一个物体位姿，并将对应的手指位置设为全空
    (3) 配置：遍历状态，当物体到达记录的位姿时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
            如果手指位置为全空，则设置后面最近的手指位置为子目标
    """

    # ********** 初选子目标 **********
    is_last_fl_contact = False
    is_last_fr_contact = False
    obj_subgoals = list()   # 物体位姿子目标
    obj_subgoals_id = list()   # 物体位姿子目标索引
    fin_subgoals_obj_init = list()  # 手指位置子目标(物体坐标系下)

    contact_thresh = fin_rad + 0.01
    
    # (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    sequence_length = state.shape[0]
    for step in range(sequence_length):
        # fingers position
        fl_pos = state[step, 7:10]
        fr_pos = state[step, 10:13]
        # 计算手指与物体是否接触, 如果手指与点云的最小距离小于阈值，认为接触
        obj_pos = state[step, 13:16]
        obj_qua = state[step, 16:20]
        # 将手指位置转到物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_pos_obj = tf.transPt(fl_pos, T_f2_f1=T_O_W)
        fr_pos_obj = tf.transPt(fr_pos, T_f2_f1=T_O_W)
        # 计算点云到手指的距离
        fl_dists = object_pcd - fl_pos_obj
        fr_dists = object_pcd - fr_pos_obj
        fl_dist = np.min(np.sqrt(np.sum(np.square(fl_dists), axis=1)))
        fr_dist = np.min(np.sqrt(np.sum(np.square(fr_dists), axis=1)))

        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh

        if step > sequence_length-Tr-1:
            is_fl_contact = True
            is_fr_contact = True

        # 记录手指接触位置
        fin_subgoals_obj_init.append(
            np.concatenate((fl_pos_obj, fr_pos_obj, [is_fl_contact,], [is_fr_contact,])))
        # 记录物体位姿
        if (is_fl_contact != is_last_fl_contact) or (is_fr_contact != is_last_fr_contact):
            obj_subgoals.append(np.concatenate((obj_pos, obj_qua)))
            obj_subgoals_id.append(step)
        
        is_last_fl_contact = is_fl_contact
        is_last_fr_contact = is_fr_contact

    # ********** 子目标精简 **********
    # (2) 过滤：当记录的两次相邻物体位姿相似时，删除前一个物体位姿，并将对应的手指位置设为None
    i = 0
    while i < len(obj_subgoals)-1:
        obj_sim = check_poses_similarity(
            obj_subgoals[i], obj_subgoals[i+1], 
            pos_th=sim_thresh[0], euler_th=sim_thresh[1])   

        if obj_sim:
            # 直到下一个物体子目标之前的手指接触都设为0
            for s in np.arange(obj_subgoals_id[i], obj_subgoals_id[i+1]):
                fin_subgoals_obj_init[s] = np.zeros((8,))
            # 删除前一个物体子目标
            obj_subgoals.pop(i)
            obj_subgoals_id.pop(i)
        else:
            # 继续对比下一个
            i+=1
    # 子目标前移一位
    fin_subgoals_obj_init.pop(0)
    fin_subgoals_obj_init.append(np.zeros((8,)))


    # ********** 子目标配置 **********
    # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
    #        如果手指位置为全空，则设置后面最近的手指位置为子目标
    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    goal_thresh = fin_rad/2
    obj_sg_id = 0   # 已达到的物体位姿子目标
    last_done = 'obj'   # obj/fin
    r = 0

    for step in range(sequence_length-Tr):
        # 手指位置
        fl_pos = state[step, 7:10]
        fr_pos = state[step, 10:13]
        # 物体位姿
        obj_pos = state[step, 13:16]
        obj_qua = state[step, 16:20]
        obj_pose = np.concatenate((obj_pos, obj_qua))
        if last_done == 'obj':
            # 检测手指是否到达子目标
            if r == max_reward:
                last_done = 'fin'
        
        if last_done == 'fin' and obj_sg_id < len(obj_subgoals_id)-1:
            # 检测物体是否到达子目标
            obj_sim = check_poses_similarity(
                obj_pose, obj_subgoals[obj_sg_id+1], 
                pos_th=sim_thresh[0], euler_th=sim_thresh[1])
            if obj_sim: 
                obj_sg_id += 1
                last_done = 'obj'

        # 设置子目标
        try:
            fin_sg_id = max(obj_subgoals_id[obj_sg_id], step)
        except:
            fin_sg_id = step
        fin_sg = fin_subgoals_obj_init[fin_sg_id]

        # 记录世界坐标下的当前时刻的子目标
        fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[6]
        fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[7]
        fin_sgs.append(np.concatenate((fl_sg, fr_sg, [fin_sg[6],], [fin_sg[7],])))        
        # 记录下一时刻的子目标
        next_obj_pos = state[step+Tr, 13:16]
        next_obj_qua = state[step+Tr, 16:20]
        next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[6]
        next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[7]
        next_fin_sgs.append(np.concatenate((next_fl_sg, next_fr_sg, [fin_sg[6],], [fin_sg[7],])))

        # 计算reward
        #* 未来n步内，有一步到达goal，就设r=max
        for n in range(1, Tr+1):
            # 目标位置
            _next_obj_pos = state[step+n, 13:16]
            _next_obj_qua = state[step+n, 16:20]
            _next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[6]
            _next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[7]
            # 手指位置(世界坐标系下)
            _next_fl_pos = state[step+n, 7:10]
            _next_fr_pos = state[step+n, 10:13]
            # 手指距离 - 欧式距离
            _next_fl_dp = np.linalg.norm(_next_fl_pos - _next_fl_sg) * fin_sg[6]
            _next_fr_dp = np.linalg.norm(_next_fr_pos - _next_fr_sg) * fin_sg[7]
            if max(_next_fl_dp, _next_fr_dp) < goal_thresh:
                r = max_reward
                break
            else:
                if reward_mode == 'only_success':
                    r = 0
                elif reward_mode == 'tanh':
                    reward_weights = 3
                    r_fl = -np.tanh(_next_fl_dp * reward_weights)
                    r_fr = -np.tanh(_next_fr_dp * reward_weights)
                    r = (r_fl+r_fr) / 3 * 2 + 1
                else:
                    raise ValueError('reward_mode must be `only_success` or `tanh`')
        
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }


def get_subgoals_stage_nonprehensile(
        raw_obs: dict,
        object_pcd: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        Tr=1,
        reward_mode='tanh'
        ):
    """
    计算手指位置子目标, 不考虑平滑

    args:
        - raw_obs: h5py dict {object_pos, object_quat, eef_pos, eef_quat, fingers_position}
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - sim_thresh: list(平移误差, 弧度误差) 计算物体位姿子目标是否达到的阈值，各任务单独设置
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """
    

    """
    (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    (2) 过滤：当记录的两次相邻物体位姿相似时(阈值很小)，删除前一个物体位姿，并将对应的手指位置设为全空
    (3) 配置：遍历状态，当物体到达记录的位姿时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
            如果手指位置为全空，则设置后面最近的手指位置为子目标
    """

    # ********** 初选子目标 **********
    is_last_fl_contact = False
    is_last_fr_contact = False
    obj_subgoals = list()   # 物体位姿子目标
    obj_subgoals_id = list()   # 物体位姿子目标索引
    fin_subgoals_obj_init = list()  # 手指位置子目标(物体坐标系下)

    contact_thresh = fin_rad + 0.01 #!!!
    
    # (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    sequence_length = raw_obs['object_pos'].shape[0]
    for step in range(sequence_length):
        # fingers position
        fl_pos = raw_obs['fingers_position'][step, :3]
        fr_pos = raw_obs['fingers_position'][step, 3:]
        # 计算手指与物体是否接触, 如果手指与点云的最小距离小于阈值，认为接触
        obj_pos = raw_obs['object_pos'][step]
        obj_qua = raw_obs['object_quat'][step]
        # 将手指位置转到物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_pos_obj = tf.transPt(fl_pos, T_f2_f1=T_O_W)
        fr_pos_obj = tf.transPt(fr_pos, T_f2_f1=T_O_W)
        # 计算点云到手指的距离
        fl_dists = object_pcd - fl_pos_obj
        fr_dists = object_pcd - fr_pos_obj
        fl_dist = np.min(np.sqrt(np.sum(np.square(fl_dists), axis=1)))
        fr_dist = np.min(np.sqrt(np.sum(np.square(fr_dists), axis=1)))

        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh
        # 记录手指接触位置
        fin_subgoals_obj_init.append(
            np.concatenate((fl_pos_obj, fr_pos_obj, [is_fl_contact,], [is_fr_contact,])))
        # 记录物体位姿
        if (is_fl_contact != is_last_fl_contact) or (is_fr_contact != is_last_fr_contact):
            obj_subgoals.append(np.concatenate((obj_pos, obj_qua)))
            obj_subgoals_id.append(step)
        
        is_last_fl_contact = is_fl_contact
        is_last_fr_contact = is_fr_contact


    # ********** 子目标精简 **********
    # (2) 过滤：当记录的两次相邻物体位姿相似时，删除前一个物体位姿，并将对应的手指位置设为None
    i = 0
    while i < len(obj_subgoals)-1:
        obj_sim = check_poses_similarity(
            obj_subgoals[i], obj_subgoals[i+1], 
            pos_th=sim_thresh[0], euler_th=sim_thresh[1])   

        if obj_sim:
            # 直到下一个物体子目标之前的手指接触都设为0
            for s in np.arange(obj_subgoals_id[i], obj_subgoals_id[i+1]):
                fin_subgoals_obj_init[s] = np.zeros((8,))
            # 删除前一个物体子目标
            obj_subgoals.pop(i)
            obj_subgoals_id.pop(i)
        else:
            # 继续对比下一个
            i+=1
    # 子目标前移一位
    fin_subgoals_obj_init.pop(0)
    fin_subgoals_obj_init.append(np.zeros((8,)))


    # ********** 子目标配置 **********
    # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
    #        如果手指位置为全空，则设置后面最近的手指位置为子目标
    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    goal_thresh = fin_rad/2
    obj_sg_id = 0   # 已达到的物体位姿子目标
    last_done = 'obj'   # obj/fin
    r = 0
    for step in range(sequence_length-Tr):
        # 手指位置
        fl_pos = raw_obs['fingers_position'][step, :3]
        fr_pos = raw_obs['fingers_position'][step, 3:]
        # 物体位姿
        obj_pos = raw_obs['object_pos'][step]
        obj_qua = raw_obs['object_quat'][step]
        obj_pose = np.concatenate((obj_pos, obj_qua))
        if last_done == 'obj':
            # 检测手指是否到达子目标
            if r == max_reward: #!
                last_done = 'fin'
        
        if last_done == 'fin' and obj_sg_id < len(obj_subgoals_id)-1:
            # 检测物体是否到达子目标
            obj_sim = check_poses_similarity(
                obj_pose, obj_subgoals[obj_sg_id+1], 
                pos_th=sim_thresh[0], euler_th=sim_thresh[1])
            if obj_sim: 
                obj_sg_id += 1
                last_done = 'obj'

        # 设置子目标
        fin_sg_id = max(obj_subgoals_id[obj_sg_id], step)
        fin_sg = fin_subgoals_obj_init[fin_sg_id]

        # 记录世界坐标下的当前时刻的子目标
        fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[6]
        fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[7]
        fin_sgs.append(np.concatenate((fl_sg, fr_sg, [fin_sg[6],], [fin_sg[7],])))        
        # 记录下一时刻的子目标
        next_obj_pos = raw_obs['object_pos'][step+Tr]
        next_obj_qua = raw_obs['object_quat'][step+Tr]
        next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[6]
        next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[7]
        next_fin_sgs.append(np.concatenate((next_fl_sg, next_fr_sg, [fin_sg[6],], [fin_sg[7],])))

        # 计算reward
        #* 未来n步内，有一步到达goal，就设r=max
        for n in range(1, Tr+1):
            # 目标位置
            _next_obj_pos = raw_obs['object_pos'][step+n]
            _next_obj_qua = raw_obs['object_quat'][step+n]
            _next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[6]
            _next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[7]
            # 手指位置(世界坐标系下)
            _next_fl_pos = raw_obs['fingers_position'][step+n, :3]
            _next_fr_pos = raw_obs['fingers_position'][step+n, 3:]
            # 手指距离 - 欧式距离
            _next_fl_dp = np.linalg.norm(_next_fl_pos - _next_fl_sg) * fin_sg[6]
            _next_fr_dp = np.linalg.norm(_next_fr_pos - _next_fr_sg) * fin_sg[7]
            if max(_next_fl_dp, _next_fr_dp) < goal_thresh:
                r = max_reward
                break
            else:
                if reward_mode == 'only_success':
                    r = 0
                elif reward_mode == 'tanh':
                    reward_weights = 3
                    r_fl = -np.tanh(_next_fl_dp * reward_weights)
                    r_fr = -np.tanh(_next_fr_dp * reward_weights)
                    r = (r_fl+r_fr) / 3 * 2 + 1
                else:
                    raise ValueError('reward_mode must be `only_success` or `tanh`')
        
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }


def get_subgoals_realtime_nonprehensile(
        raw_obs: dict,
        object_pcd: np.ndarray,
        fin_rad: float,
        max_reward=10,
        Tr=1,
        reward_mode='tanh'
        ):
    """
    计算手指位置子目标, 以后面时刻中第一个接触物体的接触点为子目标

    args:
        - raw_obs: h5py dict {object_pos, object_quat, eef_pos, eef_quat, fingers_position}
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """

    # ********** 记录接触位置 **********
    fin_subgoals_obj = list()  # 手指位置子目标(物体坐标系下)
    contact_thresh = fin_rad + 0.01
    sequence_length = raw_obs['object_pos'].shape[0]
    for step in range(sequence_length):
        # fingers position
        fl_pos = raw_obs['fingers_position'][step, :3]
        fr_pos = raw_obs['fingers_position'][step, 3:]
        # 计算手指与物体是否接触, 如果手指与点云的最小距离小于阈值，认为接触
        obj_pos = raw_obs['object_pos'][step]
        obj_qua = raw_obs['object_quat'][step]
        # 将手指位置转到物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_pos_obj = tf.transPt(fl_pos, T_f2_f1=T_O_W)
        fr_pos_obj = tf.transPt(fr_pos, T_f2_f1=T_O_W)
        # 计算点云到手指的距离
        fl_dists = object_pcd - fl_pos_obj
        fr_dists = object_pcd - fr_pos_obj
        fl_dist = np.min(np.sqrt(np.sum(np.square(fl_dists), axis=1)))
        fr_dist = np.min(np.sqrt(np.sum(np.square(fr_dists), axis=1)))

        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh
        if is_fl_contact or is_fr_contact:
            # 记录手指接触位置
            fin_subgoals_obj.append(
                np.concatenate((fl_pos_obj, fr_pos_obj, [is_fl_contact,], [is_fr_contact,])))
        else:
            fin_subgoals_obj.append(None)


    # ********** 子目标补全 **********
    for step in range(sequence_length)[:-1][::-1]:
        if fin_subgoals_obj[step] is None and fin_subgoals_obj[step+1] is not None:
            fin_subgoals_obj[step] = fin_subgoals_obj[step+1]
    fin_subgoals_obj.pop(0) # 子目标前移一位


    # ********** 子目标配置 **********
    # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
    #        如果手指位置为全空，则设置后面最近的手指位置为子目标
    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    goal_thresh = fin_rad/2
    for step in range(sequence_length-Tr):
        # 物体位姿
        obj_pos = raw_obs['object_pos'][step]
        obj_qua = raw_obs['object_quat'][step]
        # 设置子目标
        fin_sg = fin_subgoals_obj[step]

        # 记录世界坐标下的当前时刻的子目标
        fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[6]
        fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[7]
        fin_sgs.append(np.concatenate((fl_sg, fr_sg, [fin_sg[6],], [fin_sg[7],])))        
        # 记录下一时刻的子目标
        next_obj_pos = raw_obs['object_pos'][step+Tr]
        next_obj_qua = raw_obs['object_quat'][step+Tr]
        next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[6]
        next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[7]
        next_fin_sgs.append(np.concatenate((next_fl_sg, next_fr_sg, [fin_sg[6],], [fin_sg[7],])))

        # 计算reward
        #* 未来n步内，有一步到达goal，就设r=max
        for n in range(1, Tr+1):
            # 目标位置
            _next_obj_pos = raw_obs['object_pos'][step+n]
            _next_obj_qua = raw_obs['object_quat'][step+n]
            _next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[6]
            _next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[7]
            # 手指位置(世界坐标系下)
            _next_fl_pos = raw_obs['fingers_position'][step+n, :3]
            _next_fr_pos = raw_obs['fingers_position'][step+n, 3:]
            # 手指距离 - 欧式距离
            next_fl_dp = np.linalg.norm(_next_fl_pos - _next_fl_sg) * fin_sg[6]
            next_fr_dp = np.linalg.norm(_next_fr_pos - _next_fr_sg) * fin_sg[7]
            if max(next_fl_dp, next_fr_dp) < goal_thresh:
                r = max_reward
                break
            else:
                if reward_mode == 'only_success':
                    r = 0
                elif reward_mode == 'tanh':
                    reward_weights = 3
                    r_fl = -np.tanh(next_fl_dp * reward_weights)
                    r_fr = -np.tanh(next_fr_dp * reward_weights)
                    r = (r_fl+r_fr) / 3 * 2 + 1
                else:
                    raise ValueError('reward_mode must be `only_success` or `tanh`')
        
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }


def get_subgoals_stage_robomimic(
        raw_obs: dict,
        object_pcd: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        reward_mode='tanh',
        Tr=1
        ):
    """
    计算手指位置子目标, 不考虑平滑

    args:
        - raw_obs: h5py dict {object_pos, object_quat, eef_pos, eef_quat, fingers_position}
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - sim_thresh: list(平移误差, 弧度误差) 计算物体位姿子目标是否达到的阈值，各任务单独设置
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """
    

    """
    (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    (2) 过滤：当记录的两次相邻物体位姿相似时(阈值很小)，删除前一个物体位姿，并将对应的手指位置设为全空
    (3) 配置：遍历状态，当物体到达记录的位姿时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
            如果手指位置为全空，则设置后面最近的手指位置为子目标
    """

    # ********** 初选子目标 **********
    is_last_fl_contact = False
    is_last_fr_contact = False
    obj_subgoals = list()   # 物体位姿子目标
    obj_subgoals_id = list()   # 物体位姿子目标索引
    fin_subgoals_obj_init = list()  # 手指位置子目标(物体坐标系下)

    contact_thresh = fin_rad + 0.01
    
    # (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    sequence_length = raw_obs['object'].shape[0]
    for step in range(sequence_length):
        # fingers position
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0]+0.0145/2,
            raw_obs['robot0_gripper_qpos'][step, 1]-0.0145/2
            )
        # 计算手指与物体是否接触, 如果手指与点云的最小距离小于阈值，认为接触
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        # 将手指位置转到物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_pos_obj = tf.transPt(fl_pos, T_f2_f1=T_O_W)
        fr_pos_obj = tf.transPt(fr_pos, T_f2_f1=T_O_W)
        # 计算点云到手指的距离
        fl_dists = object_pcd - fl_pos_obj
        fr_dists = object_pcd - fr_pos_obj
        fl_dist = np.min(np.sqrt(np.sum(np.square(fl_dists), axis=1)))
        fr_dist = np.min(np.sqrt(np.sum(np.square(fr_dists), axis=1)))

        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh
        # 记录手指接触位置
        fin_subgoals_obj_init.append(
            np.concatenate((fl_pos_obj, fr_pos_obj, [is_fl_contact,], [is_fr_contact,])))
        # 记录物体位姿
        if (is_fl_contact != is_last_fl_contact) or (is_fr_contact != is_last_fr_contact):
            obj_subgoals.append(np.concatenate((obj_pos, obj_qua)))
            obj_subgoals_id.append(step)
        
        is_last_fl_contact = is_fl_contact
        is_last_fr_contact = is_fr_contact


    # ********** 子目标精简 **********
    # (2) 过滤：当记录的两次相邻物体位姿相似时，删除前一个物体位姿，并将对应的手指位置设为None
    i = 0
    while i < len(obj_subgoals)-1:
        obj_sim = check_poses_similarity(
            obj_subgoals[i], obj_subgoals[i+1], 
            pos_th=sim_thresh[0], euler_th=sim_thresh[1])   

        if obj_sim:
            # 直到下一个物体子目标之前的手指接触都设为0
            for s in np.arange(obj_subgoals_id[i], obj_subgoals_id[i+1]):
                fin_subgoals_obj_init[s] = np.zeros((8,))
            # 删除前一个物体子目标
            obj_subgoals.pop(i)
            obj_subgoals_id.pop(i)
        else:
            # 继续对比下一个
            i+=1
    # 子目标前移一位
    fin_subgoals_obj_init.pop(0)
    fin_subgoals_obj_init.append(np.zeros((8,)))


    # ********** 子目标配置 **********
    # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
    #        如果手指位置为全空，则设置后面最近的手指位置为子目标
    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    goal_thresh = fin_rad/2
    obj_sg_id = 0   # 已达到的物体位姿子目标
    last_done = 'obj'   # obj/fin
    r = 0
    for step in range(sequence_length-Tr):
        # 手指位置
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0]+0.0145/2,
            raw_obs['robot0_gripper_qpos'][step, 1]-0.0145/2
            )
        # 物体位姿
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        obj_pose = np.concatenate((obj_pos, obj_qua))
        if last_done == 'obj':
            # 检测手指是否到达子目标
            if r == max_reward: #!
                last_done = 'fin'
        
        if last_done == 'fin' and obj_sg_id < len(obj_subgoals_id)-1:
            # 检测物体是否到达子目标
            obj_sim = check_poses_similarity(
                obj_pose, obj_subgoals[obj_sg_id+1], 
                pos_th=sim_thresh[0], euler_th=sim_thresh[1])
            if obj_sim: 
                obj_sg_id += 1
                last_done = 'obj'

        # 设置子目标
        fin_sg_id = max(obj_subgoals_id[obj_sg_id], step)
        fin_sg = fin_subgoals_obj_init[fin_sg_id]

        # 记录世界坐标下的当前时刻的子目标
        fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[6]
        fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[7]
        fin_sgs.append(np.concatenate((fl_sg, fr_sg, [fin_sg[6],], [fin_sg[7],])))        
        # 记录下一时刻的子目标
        next_obj_pos = raw_obs['object'][step+Tr, :3]
        next_obj_qua = raw_obs['object'][step+Tr, 3:7]
        next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[6]
        next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[7]
        next_fin_sgs.append(np.concatenate((next_fl_sg, next_fr_sg, [fin_sg[6],], [fin_sg[7],])))

        # 计算reward
        #* 未来n步内，有一步到达goal，就设r=max
        for n in range(1, Tr+1):
            # 目标位置
            _next_obj_pos = raw_obs['object'][step+n, :3]
            _next_obj_qua = raw_obs['object'][step+n, 3:7]
            _next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[6]
            _next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[7]
            # 手指位置(世界坐标系下)
            _next_fl_pos, _next_fr_pos = getFingersPos(
                raw_obs['robot0_eef_pos'][step+n], 
                raw_obs['robot0_eef_quat'][step+n], 
                raw_obs['robot0_gripper_qpos'][step+n, 0]+0.0145/2,
                raw_obs['robot0_gripper_qpos'][step+n, 1]-0.0145/2
                )
            
            # 手指距离 - 欧式距离
            _next_fl_dp = np.linalg.norm(_next_fl_pos - _next_fl_sg) * fin_sg[6]
            _next_fr_dp = np.linalg.norm(_next_fr_pos - _next_fr_sg) * fin_sg[7]
            if max(_next_fl_dp, _next_fr_dp) < goal_thresh:
                r = max_reward
                break
            else:
                if reward_mode == 'only_success':
                    r = 0
                elif reward_mode == 'tanh':
                    reward_weights = 3
                    r_fl = -np.tanh(_next_fl_dp * reward_weights)
                    r_fr = -np.tanh(_next_fr_dp * reward_weights)
                    r = (r_fl+r_fr) / 3 * 2 + 1
                else:
                    raise ValueError('reward_mode must be `only_success` or `tanh`')
        
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }


def get_subgoals_stage_robomimic_v61(
        raw_obs: dict,
        object_pcd: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        reward_mode='tanh',
        horizon=16
        ):
    """
    计算手指位置子目标, 不考虑平滑

    计算state之后horizon个状态的next_subgoal和reward

    args:
        - raw_obs: h5py dict {object_pos, object_quat, eef_pos, eef_quat, fingers_position}
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - sim_thresh: list(平移误差, 弧度误差) 计算物体位姿子目标是否达到的阈值，各任务单独设置
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'

    return:
        - subgoal: (N, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N, horizon, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N, horizon)
    """

    """
    (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    (2) 过滤：当记录的两次相邻物体位姿相似时(阈值很小)，删除前一个物体位姿，并将对应的手指位置设为全空
    (3) 配置：遍历状态，当物体到达记录的位姿时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
            如果手指位置为全空，则设置后面最近的手指位置为子目标
    """

    # ********** 初选子目标 **********
    is_last_fl_contact = False
    is_last_fr_contact = False
    obj_subgoals = list()   # 物体位姿子目标
    obj_subgoals_id = list()   # 物体位姿子目标索引
    fin_subgoals_obj_init = list()  # 手指位置子目标(物体坐标系下)

    contact_thresh = fin_rad + 0.01
    
    # (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    sequence_length = raw_obs['object'].shape[0]
    for step in range(sequence_length):
        # fingers position
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0]+0.0145/2,
            raw_obs['robot0_gripper_qpos'][step, 1]-0.0145/2
            )
        # 计算手指与物体是否接触, 如果手指与点云的最小距离小于阈值，认为接触
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        # 将手指位置转到物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_pos_obj = tf.transPt(fl_pos, T_f2_f1=T_O_W)
        fr_pos_obj = tf.transPt(fr_pos, T_f2_f1=T_O_W)
        # 计算点云到手指的距离
        fl_dists = object_pcd - fl_pos_obj
        fr_dists = object_pcd - fr_pos_obj
        fl_dist = np.min(np.sqrt(np.sum(np.square(fl_dists), axis=1)))
        fr_dist = np.min(np.sqrt(np.sum(np.square(fr_dists), axis=1)))

        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh
        # 记录手指接触位置
        fin_subgoals_obj_init.append(
            np.concatenate((fl_pos_obj, fr_pos_obj, [is_fl_contact,], [is_fr_contact,])))
        # 记录物体位姿
        if (is_fl_contact != is_last_fl_contact) or (is_fr_contact != is_last_fr_contact):
            obj_subgoals.append(np.concatenate((obj_pos, obj_qua)))
            obj_subgoals_id.append(step)
        
        is_last_fl_contact = is_fl_contact
        is_last_fr_contact = is_fr_contact


    # ********** 子目标精简 **********
    # (2) 过滤：当记录的两次相邻物体位姿相似时，删除前一个物体位姿，并将对应的手指位置设为None
    i = 0
    while i < len(obj_subgoals)-1:
        obj_sim = check_poses_similarity(
            obj_subgoals[i], obj_subgoals[i+1], 
            pos_th=sim_thresh[0], euler_th=sim_thresh[1])   

        if obj_sim:
            # 直到下一个物体子目标之前的手指接触都设为0
            for s in np.arange(obj_subgoals_id[i], obj_subgoals_id[i+1]):
                fin_subgoals_obj_init[s] = np.zeros((8,))
            # 删除前一个物体子目标
            obj_subgoals.pop(i)
            obj_subgoals_id.pop(i)
        else:
            # 继续对比下一个
            i+=1
    # 子目标前移一位
    fin_subgoals_obj_init.pop(0)
    fin_subgoals_obj_init.append(np.zeros((8,)))


    # ********** 子目标配置 **********
    # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
    #        如果手指位置为全空，则设置后面最近的手指位置为子目标
    fin_sgs = list()
    next_fin_sgss = list()
    rewards = list()
    goal_thresh = fin_rad     #!!! 增加 原来是fin_rad/2
    # goal_thresh = 0.03     #!!! 增加 原来是fin_rad/2
    obj_sg_id = 0   # 已达到的物体位姿子目标
    last_done = 'obj'   # obj/fin
    r = 0
    for step in range(sequence_length):
        # 手指位置
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0]+0.0145/2,
            raw_obs['robot0_gripper_qpos'][step, 1]-0.0145/2
            )
        # 物体位姿
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        obj_pose = np.concatenate((obj_pos, obj_qua))
        if last_done == 'obj':
            # 检测手指是否到达子目标
            if r == max_reward: #!
                last_done = 'fin'
        
        if last_done == 'fin' and obj_sg_id < len(obj_subgoals_id)-1:
            # 检测物体是否到达子目标
            obj_sim = check_poses_similarity(
                obj_pose, obj_subgoals[obj_sg_id+1], 
                pos_th=sim_thresh[0], euler_th=sim_thresh[1])
            if obj_sim: 
                obj_sg_id += 1
                last_done = 'obj'

        # 设置子目标
        fin_sg_id = max(obj_subgoals_id[obj_sg_id], step)
        fin_sg = fin_subgoals_obj_init[fin_sg_id]

        # 记录世界坐标下的当前时刻的子目标
        fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[6]
        fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[7]
        fin_sgs.append(np.concatenate((fl_sg, fr_sg, [fin_sg[6],], [fin_sg[7],]))) 

        # 记录下一时刻的子目标
        next_fin_sgs = list()
        reward = list()
        for h in range(horizon+1)[1:]:
            next_step = step+h
            if next_step >= sequence_length:
                next_fin_sgs.append(fin_sgs[-1])
            else:
                next_obj_pos = raw_obs['object'][next_step, :3]
                next_obj_qua = raw_obs['object'][next_step, 3:7]
                next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[6]
                next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[7]
                next_fin_sgs.append(np.concatenate((next_fl_sg, next_fr_sg, [fin_sg[6],], [fin_sg[7],])))

            # 计算reward
            # 手指位置(世界坐标系下)
            if next_step >= sequence_length:
                r = max_reward
            else:
                next_fl_pos, next_fr_pos = getFingersPos(
                    raw_obs['robot0_eef_pos'][next_step], 
                    raw_obs['robot0_eef_quat'][next_step], 
                    raw_obs['robot0_gripper_qpos'][next_step, 0]+0.0145/2,
                    raw_obs['robot0_gripper_qpos'][next_step, 1]-0.0145/2)
                
                # 手指距离 - 欧式距离
                next_fl_dp = np.linalg.norm(next_fl_pos - next_fin_sgs[-1][:3]) * next_fin_sgs[-1][6]
                next_fr_dp = np.linalg.norm(next_fr_pos - next_fin_sgs[-1][3:6]) * next_fin_sgs[-1][7]
                if max(next_fl_dp, next_fr_dp) < goal_thresh:
                    r = max_reward
                else:
                    if reward_mode == 'only_success':
                        r = 0
                    elif reward_mode == 'tanh':
                        reward_weights = 3
                        r_fl = -np.tanh(next_fl_dp * reward_weights)
                        r_fr = -np.tanh(next_fr_dp * reward_weights)
                        r = (r_fl+r_fr) / 3 * 2 + 1
                    else:
                        raise ValueError('reward_mode must be `only_success` or `tanh`')
            
            reward.append(r)
        
        next_fin_sgss.append(np.array(next_fin_sgs))
        rewards.append(np.array(reward))

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgss),
        'reward': np.array(rewards)
    }


def get_subgoals_realtime_robomimic(
        raw_obs: dict,
        object_pcd: np.ndarray,
        fin_rad: float,
        max_reward=10,
        reward_mode='tanh',
        Tr=1
        ):
    """
    计算手指位置子目标, 不考虑平滑

    args:
        - raw_obs: h5py dict {object_pos, object_quat, eef_pos, eef_quat, fingers_position}
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """


    # ********** 初选子目标 **********
    fin_subgoals_obj = list()  # 手指位置子目标(物体坐标系下)
    contact_thresh = fin_rad + 0.01
    sequence_length = raw_obs['object'].shape[0]
    for step in range(sequence_length):
        # fingers position
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0]+0.0145/2,
            raw_obs['robot0_gripper_qpos'][step, 1]-0.0145/2
            )
        # 计算手指与物体是否接触, 如果手指与点云的最小距离小于阈值，认为接触
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        # 将手指位置转到物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_pos_obj = tf.transPt(fl_pos, T_f2_f1=T_O_W)
        fr_pos_obj = tf.transPt(fr_pos, T_f2_f1=T_O_W)
        # 计算点云到手指的距离
        fl_dists = object_pcd - fl_pos_obj
        fr_dists = object_pcd - fr_pos_obj
        fl_dist = np.min(np.sqrt(np.sum(np.square(fl_dists), axis=1)))
        fr_dist = np.min(np.sqrt(np.sum(np.square(fr_dists), axis=1)))

        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh
        fin_subgoals_obj.append(
            np.concatenate((fl_pos_obj, fr_pos_obj, [is_fl_contact,], [is_fr_contact,])))

    fin_subgoals_obj.pop(0) # 子目标前移一位


    # ********** 子目标配置 **********
    # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
    #        如果手指位置为全空，则设置后面最近的手指位置为子目标
    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    goal_thresh = fin_rad/2
    for step in range(sequence_length-Tr):
        # 物体位姿
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        obj_pose = np.concatenate((obj_pos, obj_qua))
        # 设置子目标
        fin_sg = fin_subgoals_obj[step]

        # 记录世界坐标下的当前时刻的子目标
        fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[6]
        fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[7]
        fin_sgs.append(np.concatenate((fl_sg, fr_sg, [fin_sg[6],], [fin_sg[7],])))        
        # 记录下一时刻的子目标
        next_obj_pos = raw_obs['object'][step+Tr, :3]
        next_obj_qua = raw_obs['object'][step+Tr, 3:7]
        next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[6]
        next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[7]
        next_fin_sgs.append(np.concatenate((next_fl_sg, next_fr_sg, [fin_sg[6],], [fin_sg[7],])))

        # 计算reward
        #* 未来n步内，有一步到达goal，就设r=max
        for n in range(1, Tr+1):
            # 目标位置
            _next_obj_pos = raw_obs['object'][step+n, :3]
            _next_obj_qua = raw_obs['object'][step+n, 3:7]
            _next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[6]
            _next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[7]
            # 手指位置(世界坐标系下)
            next_fl_pos, next_fr_pos = getFingersPos(
                raw_obs['robot0_eef_pos'][step+n], 
                raw_obs['robot0_eef_quat'][step+n], 
                raw_obs['robot0_gripper_qpos'][step+1, 0]+0.0145/2,
                raw_obs['robot0_gripper_qpos'][step+1, 1]-0.0145/2
                )
            
            # 手指距离 - 欧式距离
            next_fl_dp = np.linalg.norm(next_fl_pos - _next_fl_sg) * fin_sg[6]
            next_fr_dp = np.linalg.norm(next_fr_pos - _next_fr_sg) * fin_sg[7]
            if max(next_fl_dp, next_fr_dp) < goal_thresh:
                r = max_reward
                break
            else:
                if reward_mode == 'only_success':
                    r = 0
                elif reward_mode == 'tanh':
                    reward_weights = 3
                    r_fl = -np.tanh(next_fl_dp * reward_weights)
                    r_fr = -np.tanh(next_fr_dp * reward_weights)
                    r = (r_fl+r_fr) / 3 * 2 + 1
                else:
                    raise ValueError('reward_mode must be `only_success` or `tanh`')
        
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }

def get_subgoals_pusht(
        raw_obs: np.ndarray,
        episode_ends: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        Tr=1,
        reward_mode='tanh'
        ):
    """
    计算手指位置子目标, 不考虑平滑
    输入包含多个轨迹, 根据物体位姿突变划分轨迹

    args:
        - raw_obs: (N, 5) 手指位置2/物体位置2/旋转角1 （旋转角为0时，T的竖线朝上，物体逆时针旋转时，角度增加）
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - sim_thresh: list(平移误差, 弧度误差) 计算物体位姿子目标是否达到的阈值，各任务单独设置

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """
    # 构建物体点云
    object_pcd = create_pusht_pts(pts_num=1024*5) # (n, 2)

    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    episode_ends = np.insert(episode_ends, 0, 0)
    for j in range(episode_ends.shape[0])[1:]:
        start = episode_ends[j-1]
        end = episode_ends[j]

        # ********** 初选子目标 **********
        is_last_fin_contact = False
        obj_subgoals = list()   # 物体位姿子目标
        obj_subgoals_id = list()   # 物体位姿子目标索引
        fin_subgoals_obj_init = list()  # 手指位置子目标(物体坐标系下)
        contact_thresh = fin_rad + 3
        
        # (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
        for step in np.arange(start, end):
            # fingers position
            fin_pos = raw_obs[step, :2]
            obj_pos = raw_obs[step, 2:4]
            obj_rad = raw_obs[step, 4]
            # 将手指位置转到物体坐标系
            T_W_O = tf.PosRad_to_Tmat(obj_pos, obj_rad)
            T_O_W = np.linalg.inv(T_W_O)
            fin_pos_obj = tf.transPt2D(fin_pos, T_f2_f1=T_O_W)
            # 计算点云到手指的距离
            fin_dists = object_pcd - fin_pos_obj
            fin_dist = np.min(np.sqrt(np.sum(np.square(fin_dists), axis=1)))
            is_fin_contact = fin_dist < contact_thresh
            # 记录手指接触位置
            fin_subgoals_obj_init.append(np.append(fin_pos_obj, float(is_fin_contact)))
            # 记录物体位姿
            if is_fin_contact != is_last_fin_contact:
                obj_subgoals.append(raw_obs[step, 2:])
                obj_subgoals_id.append(step-start)
            
            is_last_fin_contact = is_fin_contact

        # ********** 子目标精简 **********
        # (2) 过滤：当记录的两次相邻物体位姿相似时，删除前一个物体位姿，并将对应的手指位置设为None
        i = 0
        while i < len(obj_subgoals)-1:
            obj_sim = check_poses_similarity_2d(
                obj_subgoals[i], obj_subgoals[i+1], 
                pos_th=sim_thresh[0], euler_th=sim_thresh[1])   

            if obj_sim:
                # 直到下一个物体子目标之前的手指接触都设为0
                for s in np.arange(obj_subgoals_id[i], obj_subgoals_id[i+1]):
                    fin_subgoals_obj_init[s] = np.zeros((3,))
                # 删除前一个物体子目标
                obj_subgoals.pop(i)
                obj_subgoals_id.pop(i)
            else:
                # 继续对比下一个
                i+=1
        # 子目标前移一位
        fin_subgoals_obj_init.pop(0)
        fin_subgoals_obj_init.append(np.zeros((3,)))

        # ********** 子目标配置 **********
        # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
        #        如果手指位置为全空，则设置后面最近的手指位置为子目标
        goal_thresh = fin_rad/2
        obj_sg_id = 0   # 已达到的物体位姿子目标
        last_done = 'obj'   # obj/fin
        r = 0
        for step in np.arange(start, end):
            if step >= end-Tr:
                fin_sgs.append(np.zeros((3,)))
                next_fin_sgs.append(np.zeros((3,)))
                reward.append(max_reward)
                if step == end-1: break
                else: continue

            # 物体位姿
            obj_pose = raw_obs[step, 2:]
            if last_done == 'obj':
                # 检测手指是否到达子目标
                if r == max_reward:
                    last_done = 'fin'
            
            if last_done == 'fin' and obj_sg_id < len(obj_subgoals_id)-1:
                # 检测物体是否到达子目标
                obj_sim = check_poses_similarity_2d(
                    obj_pose, obj_subgoals[obj_sg_id+1], 
                    pos_th=sim_thresh[0], euler_th=sim_thresh[1])
                if obj_sim: 
                    obj_sg_id += 1
                    last_done = 'obj'

            # 设置子目标
            fin_sg_id = max(obj_subgoals_id[obj_sg_id], step-start)
            fin_sg = fin_subgoals_obj_init[fin_sg_id]

            # 记录世界坐标下的当前时刻的子目标
            T_W_O = tf.PosRad_to_Tmat(raw_obs[step, 2:4], raw_obs[step, 4])
            fin_sg_pos = tf.transPt2D(fin_sg[:2], T_W_O) * fin_sg[2]
            fin_sgs.append(np.append(fin_sg_pos, fin_sg[2]))
            # 记录下一时刻的子目标
            T_W_On = tf.PosRad_to_Tmat(raw_obs[step+Tr, 2:4], raw_obs[step+Tr, 4])
            next_fin_sg_pos = tf.transPt2D(fin_sg[:2], T_W_On) * fin_sg[2]
            next_fin_sgs.append(np.append(next_fin_sg_pos, fin_sg[2]))

            # 计算reward
            #* 未来n步内，有一步到达goal，就设r=max
            for n in range(1, Tr+1):
                # 手指位置(世界坐标系下)
                _next_fin_pos = raw_obs[step+n, :2]
                # 目标位置
                _T_W_On = tf.PosRad_to_Tmat(raw_obs[step+n, 2:4], raw_obs[step+n, 4])
                _next_fin_sg_pos = tf.transPt2D(fin_sg[:2], _T_W_On) * fin_sg[2]
                # 手指距离 - 欧式距离
                _next_fin_dp = np.linalg.norm(_next_fin_pos - _next_fin_sg_pos) * fin_sg[2]
                if _next_fin_dp < goal_thresh:
                    r = max_reward
                    break
                else:
                    if reward_mode == 'only_success':
                        r = 0
                    elif reward_mode == 'tanh':
                        reward_weights = 3
                        r_fl = -np.tanh(_next_fin_dp/15*0.008 * reward_weights)
                        r = r_fl / 3 * 2 + 1
                    else:
                        raise ValueError('reward_mode must be `only_success` or `tanh`')
                
            reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }


def get_subgoals_realtime_pusht(
        raw_obs: np.ndarray,
        episode_ends: np.ndarray,
        fin_rad: float,
        max_reward=10,
        Tr=1,
        reward_mode='tanh'
        ):
    """
    计算手指位置子目标, 不考虑平滑
    输入包含多个轨迹, 根据物体位姿突变划分轨迹

    args:
        - raw_obs: (N, 5) 手指位置2/物体位置2/旋转角1 （旋转角为0时，T的竖线朝上，物体逆时针旋转时，角度增加）
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - sim_thresh: list(平移误差, 弧度误差) 计算物体位姿子目标是否达到的阈值，各任务单独设置

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """
    # 构建物体点云
    object_pcd = create_pusht_pts(pts_num=1024*5) # (n, 2)

    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    episode_ends = np.insert(episode_ends, 0, 0)
    for j in range(episode_ends.shape[0])[1:]:
        start = episode_ends[j-1]
        end = episode_ends[j]

        # ********** 初选子目标 **********
        fin_subgoals_obj_init = list()  # 手指位置子目标(物体坐标系下)
        contact_thresh = fin_rad + 3
        
        # (1) 初选：当手指与物体接触时记录手指位置
        for step in np.arange(start, end):
            # fingers position
            fin_pos = raw_obs[step, :2]
            obj_pos = raw_obs[step, 2:4]
            obj_rad = raw_obs[step, 4]
            # 将手指位置转到物体坐标系
            T_W_O = tf.PosRad_to_Tmat(obj_pos, obj_rad)
            T_O_W = np.linalg.inv(T_W_O)
            fin_pos_obj = tf.transPt2D(fin_pos, T_f2_f1=T_O_W)
            # 计算点云到手指的距离
            fin_dists = object_pcd - fin_pos_obj
            fin_dist = np.min(np.sqrt(np.sum(np.square(fin_dists), axis=1)))
            is_fin_contact = fin_dist < contact_thresh
            # print('is_fin_contact =', is_fin_contact)
            # 记录手指接触位置
            fin_subgoals_obj_init.append(np.append(fin_pos_obj, float(is_fin_contact)))

        # 子目标前移一位
        fin_subgoals_obj_init.pop(0)

        # ********** 子目标配置 **********
        goal_thresh = fin_rad/2
        r = 0
        for step in np.arange(start, end):
            if step >= end-Tr:
                fin_sgs.append(np.zeros((3,)))
                next_fin_sgs.append(np.zeros((3,)))
                reward.append(max_reward)
                if step == end-1: break
                else: continue

            # 设置子目标
            fin_sg = fin_subgoals_obj_init[step - start]

            # 记录世界坐标下的当前时刻的子目标
            T_W_O = tf.PosRad_to_Tmat(raw_obs[step, 2:4], raw_obs[step, 4])
            fin_sg_pos = tf.transPt2D(fin_sg[:2], T_W_O) * fin_sg[2]
            fin_sgs.append(np.append(fin_sg_pos, fin_sg[2]))
            # 记录下一时刻的子目标
            T_W_On = tf.PosRad_to_Tmat(raw_obs[step+Tr, 2:4], raw_obs[step+Tr, 4])
            next_fin_sg_pos = tf.transPt2D(fin_sg[:2], T_W_On) * fin_sg[2]
            next_fin_sgs.append(np.append(next_fin_sg_pos, fin_sg[2]))

            # 计算reward
            #* 未来n步内，有一步到达goal，就设r=max
            for n in range(1, Tr+1):
                # 手指位置(世界坐标系下)
                _next_fin_pos = raw_obs[step+n, :2]
                # 目标位置
                _T_W_On = tf.PosRad_to_Tmat(raw_obs[step+n, 2:4], raw_obs[step+n, 4])
                _next_fin_sg_pos = tf.transPt2D(fin_sg[:2], _T_W_On) * fin_sg[2]
                # 手指距离 - 欧式距离
                _next_fin_dp = np.linalg.norm(_next_fin_pos - _next_fin_sg_pos) * fin_sg[2]
                if _next_fin_dp < goal_thresh:
                    r = max_reward
                    break
                else:
                    if reward_mode == 'only_success':
                        r = 0
                    elif reward_mode == 'tanh':
                        reward_weights = 3
                        r_fl = -np.tanh(_next_fin_dp/15*0.008 * reward_weights)
                        r = r_fl / 3 * 2 + 1
                    else:
                        raise ValueError('reward_mode must be `only_success` or `tanh`')
                
            reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }


def get_subgoals_stage_real(
        state: np.ndarray,
        fin_rad,
        contact_state: np.ndarray,
        max_reward=10,
        reward_mode='tanh',
        Tr=1
        ):
    """
    计算真实任务中的手指位置子目标, 不考虑平滑

    args:
        - state (np.ndarray): (N, 13) eef_pos, eef_qua, fl_pos, fr_pos
        - contact_state (np.ndarray): (N) 手指与物体的接触状态, 0-无接触, 1-有接触
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'
        - Tr: 选择下个状态的间隔

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """
    # ********** set subgoal **********
    finger_sgs = list()  # 手指位置子目标    
    sequence_length = state.shape[0]
    for step in range(sequence_length):
        is_contact = contact_state[step]
        # fingers position
        fl_pos = state[step, 7:10]*is_contact
        fr_pos = state[step, 10:13]*is_contact
        # 记录手指接触位置
        finger_sgs.append(
            np.concatenate((fl_pos, fr_pos, [is_contact,], [is_contact,])))
    
    # 补全接触状态
    for step in range(sequence_length)[:-1][::-1]:
        if np.sum(finger_sgs[step][-2:]) == 0 and np.sum(finger_sgs[step+1][-2:]) != 0:
            finger_sgs[step] = finger_sgs[step+1]


    # ********** set next_subgoal and reward **********
    finger_sgs = np.array(finger_sgs)
    fin_sgs = finger_sgs[:-Tr]
    next_fin_sgs = finger_sgs[Tr:]
    reward = list()
    goal_thresh = fin_rad/2
    
    for step in range(sequence_length-Tr):
        # 未来Tr步内，有一步到达目标就设r=max
        for n in range(1, Tr+1):
            # 手指位置
            next_fl_pos = state[step+n, 7:10]
            next_fr_pos = state[step+n, 10:13]
            # 目标
            next_pos_sg = finger_sgs[step+n]
            # 手指距离 - 欧式距离
            next_fl_dp = np.linalg.norm(next_fl_pos - next_pos_sg[:3]) * next_pos_sg[6]
            next_fr_dp = np.linalg.norm(next_fr_pos - next_pos_sg[3:6]) * next_pos_sg[7]
            if max(next_fl_dp, next_fr_dp) < goal_thresh:
                r = max_reward
                break
            else:
                if reward_mode == 'only_success':
                    r = 0
                elif reward_mode == 'tanh':
                    reward_weights = 3
                    r_fl = -np.tanh(next_fl_dp * reward_weights)
                    r_fr = -np.tanh(next_fr_dp * reward_weights)
                    r = (r_fl+r_fr) / 3 * 2 + 1
                else:
                    raise ValueError('reward_mode must be `only_success` or `tanh`')
        
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }

def compute_reward_nextSubgoal_from_subgoal(
        subgoal: np.ndarray, 
        obj_pose: np.ndarray, 
        next_obj_pose: np.ndarray, 
        next_fin_pos: np.ndarray,
        fin_rad,
        max_reward=10,
        ) -> torch.Tensor:
    """使用新生成的subgoal计算reward
    将subgoal转到物体坐标系下，再转到下一时刻的世界坐标系下，计算subgoal与手指位置的差异

    args(torch.Tensor): 
        - subgoal: (B, 8) 当前时刻的子目标(world)
        - obj_pose: (B, 7) 当前时刻的物体位姿
        - next_obj_pose: (B, 7) 下一时刻的物体位姿
        - next_fin_pos: (B, 6) 下一时刻的手指位置
        - fin_rad: 手指半径
    
    return:
        - reward: (B,) done=1，其余为0
        - next_subgoal: (B, 8)
    """
    reward = list()
    next_subgoal = list()
    for i in range(subgoal.shape[0]):
        sg = subgoal[i]
        op = obj_pose[i]
        nop = next_obj_pose[i]
        nfp = next_fin_pos[i]
        # 子目标转到下一时刻的世界坐标系
        # (1) 转到物体坐标系: P_O_sg = T_O_W * P_W_sg
        T_W_O = tf.PosQua_to_TransMat(op[:3], op[3:])
        T_O_W = np.linalg.inv(T_W_O)
        P_O_sgl = tf.transPt(P_f1_pt=sg[:3], T_f2_f1=T_O_W)
        P_O_sgr = tf.transPt(P_f1_pt=sg[3:6], T_f2_f1=T_O_W)
        # (2) 转到下一时刻的世界坐标系: P_W_sg = T_W_O * P_O_sg
        T_W_O_ = tf.PosQua_to_TransMat(nop[:3], nop[3:])
        P_W_sgl = tf.transPt(P_f1_pt=P_O_sgl, T_f2_f1=T_W_O_) * sg[6]
        P_W_sgr = tf.transPt(P_f1_pt=P_O_sgr, T_f2_f1=T_W_O_) * sg[7]
        # 计算reward
        fl_dist = np.linalg.norm(nfp[:3] - P_W_sgl) * sg[6]
        fr_dist = np.linalg.norm(nfp[3:6] - P_W_sgr) * sg[7]
        if max(fl_dist, fl_dist) < fin_rad/2:
            r = max_reward
        else:
            reward_weights = 3
            r_fl = -np.tanh(fl_dist * reward_weights)
            r_fr = -np.tanh(fr_dist * reward_weights)
            r = (r_fl+r_fr) / 3 * 2 + 1

        reward.append(r)
        next_subgoal.append(np.concatenate((P_W_sgl, P_W_sgr, sg[6:])))

    return torch.tensor(reward), torch.tensor(np.array(next_subgoal))


def distTwoPtWithCube(rect, pt1, pt2, z_th):
    """
    计算空间中两个点的路径距离, 两点可能被矩形分割

    计算流程:
        (1) 将矩形和两个点的x维度删除
        (2) 判断两点形成的线段是否可能rect分割为两个多边形
        (3) 如果分割结果为1个多边形(即无法分割), 则直接返回pt1和pt2的L2范数
        (4) 如果分割结果为2个多边形:
        (5) 计算不含 z<(z_th+0.01) 的多边形的边长,边长不包含分割线
        (6) 计算pt1和pt2到两个分割点的距离的较小值的和, 加上第5步多边形的边长, 和x距离计算L2范数, 返回

    args:
        rect (np.array): 空间中的矩形, shape=(4,3)
        pt1 (np.array): 三维点1, shape=(3,)
        pt2 (np.array): 三维点2, shape=(3,)
        z_th (float): z坐标阈值, 路径不能在z_th下面
    """
    # (1) 将矩形和两个点的x维度删除
    rect_yz = rect[:, 1:]
    pt1_yz = pt1[1:]
    pt2_yz = pt2[1:]
    # (2) 判断两点形成的线段是否可能rect分割为两个多边形
    line = shapely.geometry.LineString([list(pt1_yz), list(pt2_yz)])
    polygon = shapely.geometry.Polygon([list(rect_yz[0]), list(rect_yz[1]), list(rect_yz[2]), list(rect_yz[3])])
    polygons = shapely.ops.split(polygon, line)
    # (3) 如果分割结果为1个多边形(即无法分割), 则直接返回pt1和pt2的L2范数
    if len(polygons.geoms) == 1:
        return np.linalg.norm(pt1 - pt2)
    # (5) 计算不含 z<(z_th+0.01) 的多边形的边长,边长不包含分割线
    id = 0
    for z in polygons.geoms[id].exterior.coords.xy[1]:
        if z < (z_th+0.01):
            id = 1
    polygon_path = polygons.geoms[id]
    ys = polygon_path.exterior.coords.xy[0]
    zs = polygon_path.exterior.coords.xy[1]
    # 计算边长, 不含分割线
    l = 0
    for i in range(len(ys)-2):
        l += math.sqrt((ys[i] - ys[i+1])**2 + (zs[i] - zs[i+1])**2)
    # print('l =', l)
    # (6) 计算pt1和pt2到两个分割点的距离的较小值的和, 加上第5步多边形的边长, 和x距离计算L2范数, 返回
    seg_pt1 = np.array([ys[-1], zs[-1]])
    seg_pt2 = np.array([ys[-2], zs[-2]])
    l1 = min( np.linalg.norm(pt1_yz - seg_pt1), np.linalg.norm(pt1_yz - seg_pt2) )
    l2 = min( np.linalg.norm(pt2_yz - seg_pt1), np.linalg.norm(pt2_yz - seg_pt2) )
    l += l1 + l2 + abs(pt1[0] - pt2[0])
    # print('l1 =', l1)
    # print('l2 =', l2)
    # print('abs(pt1[0] - pt2[0]) =', abs(pt1[0] - pt2[0]))
    return l


def getCubeXPosWorld(cube_half_size, cube_pos, cube_quat):
    """
    获取物体x轴正方向上四个角点在世界坐标系下的位置
    """
    l1, l2, l3 = cube_half_size
    pts = np.array([
            [l1, -l2, l3],
            [l1, l2, l3],
            [l1, l2, -l3],
            [l1, -l2, -l3]
        ])
    # 转到世界坐标系下
    # P_o_p
    one = np.ones((1, pts.shape[0]))
    P_O_p = np.concatenate((pts.T, one), axis=0)   # (4,4)
    # T_w_o
    cube_rotMat = tf.quaternion_to_rotation_matrix(cube_quat)
    T_W_O = tf.PosRmat_to_TransMat(cube_pos, cube_rotMat)
    # P_w_p = T_w_o * P_o_p
    P_w_p = np.matmul(T_W_O, P_O_p)
    return P_w_p.T[:, :3]


def angle_diff(angle_1, angle_2):
    """
    输入弧度，输出差值 0-pi
    """
    angle_1 = angle_1 % (2*np.pi)
    angle_2 = angle_2 % (2*np.pi)
    
    angle_min = min(angle_1, angle_2)
    angle_max = max(angle_1, angle_2)

    error = angle_max - angle_min
    if error <= np.pi:
        return error
    else:
        return angle_min + 2*np.pi - angle_max


def check_poses_similarity_moveT(pose1, pose2, pos_th=0.005, euler_th=5./180.*np.pi):
    """
    计算两位姿的相似性(旋转只计算z轴旋转)
    pose: 平移+四元数xyzw
    return: 是否相似
    """
    # 当前时刻的平移和旋转差
    obj_dp = np.linalg.norm(pose1[:3] - pose2[:3])
    # 方案1：计算欧拉角的夹角，有时候会计算错误
    # obj_euler_1 = R.from_quat(pose1[3:]).as_euler('xyz', degrees=False)
    # obj_euler_2 = R.from_quat(pose2[3:]).as_euler('xyz', degrees=False)
    # obj_dr = ([tf.angle_diff(obj_euler_1[i], obj_euler_2[i]) for i in range(3)])
    # if obj_dp > pos_th or max(obj_dr) > euler_th:
    #     return False
    
    # 方案2：计算四元数的夹角
    rz1 = tf.Qua_to_Euler(pose1[3:])[2]
    rz2 = tf.Qua_to_Euler(pose2[3:])[2]
    obj_dr = angle_diff(rz1, rz2)
    if obj_dp > pos_th or obj_dr > euler_th:
        return False

    return True

def check_poses_similarity(pose1, pose2, pos_th=0.005, euler_th=5./180.*np.pi):
    """
    计算两位姿的相似性
    pose: 平移+四元数xyzw
    return: 是否相似
    """
    # 当前时刻的平移和旋转差
    obj_dp = np.linalg.norm(pose1[:3] - pose2[:3])
    # 方案1：计算欧拉角的夹角，有时候会计算错误
    # obj_euler_1 = R.from_quat(pose1[3:]).as_euler('xyz', degrees=False)
    # obj_euler_2 = R.from_quat(pose2[3:]).as_euler('xyz', degrees=False)
    # obj_dr = ([tf.angle_diff(obj_euler_1[i], obj_euler_2[i]) for i in range(3)])
    # if obj_dp > pos_th or max(obj_dr) > euler_th:
    #     return False
    
    # 方案2：计算四元数的夹角
    obj_dr = tf.qua_diff(pose1[3:], pose2[3:])
    if obj_dp > pos_th or obj_dr > euler_th:
        return False

    return True


def check_poses_similarity_2d(pose1, pose2, pos_th=0.005, euler_th=5./180.*np.pi):
    """
    计算两位姿的相似性
    pose: 坐标+旋转角xyr
    return: 是否相似
    """
    obj_dp = np.linalg.norm(pose1[:2] - pose2[:2])
    obj_dr = angle_diff(pose1[2], pose2[2])
    if obj_dp > pos_th or obj_dr > euler_th:
        return False
    return True


def check_pos_similarity(pos1, pos2, pos_th=0.005):
    """
    计算两位姿的相似性
    pose: 平移+四元数xyzw
    return: 是否相似
    """
    # 当前时刻的平移和旋转差
    obj_dp = np.linalg.norm(pos1 - pos2)
    if obj_dp > pos_th:
        return False
    return True


def get_scene_object_pcd_goal(dataset_path, visual=False, dtype=np.float32):
    """
    get scene_pcd / object_pcd / object_goal_pose(not use in HDP)
    dataset_path: robomimic数据集路径
    visual: 是否可视化图像/pcd
    return:
        - scene_pcd: (N, 3)
        - object_pcd: (N, 3)
        - object_goal_pose: (7,) pos+quat
    """

    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map
    import time
    import hiera_diffusion_policy.common.transformation as tf

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    options = {}

    options["env_name"] = env_meta['env_name']
    options["robots"] = env_meta['env_kwargs']["robots"]
    controller_name = "OSC_POSE"
    camera = "agentview"
    segmentation_level = 'instance'  # Options are {instance, class, element}

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_names=camera,
        camera_segmentations=segmentation_level,
        camera_depths=True,
        camera_heights=512,
        camera_widths=512,
    )
    env.reset()
    
    # **************** object goal pose ****************
    object_goal_pose = env.object_goal_pose()

    # **************** image ****************
    env.remove_all_objects()    # 移除所有物体
    for i in range(50):
        action = [-1, 0, 0, 0, 0, 0, 0]
        obs, reward, done, _ = env.step(action)

    # segmentation
    img_seg = obs[f"{camera}_segmentation_{segmentation_level}"].squeeze(-1)[::-1]
    img_seg[np.where(img_seg > 0)] = 1
    # rgb
    img_rgb = obs[f"{camera}_image"][::-1]
    # depth
    img_dep = obs[f"{camera}_depth"].squeeze(-1)[::-1]
    img_dep = get_real_depth_map(env.sim, img_dep)
    if visual:
        # show
        show_rgb_seg_dep(camera, img_rgb, img_seg, img_dep)

    # **************** scene_pcd ****************
    cameraInMatrix = get_camera_intrinsic_matrix(env.sim, camera, 512, 512)
    cameraPoseMatrix = get_camera_extrinsic_matrix(env.sim, camera)
    mask = np.zeros(img_seg.shape[:2], dtype=np.bool)
    mask[np.where(img_seg == 0)] = 1
    scene_pcd = tf.create_point_cloud(img_rgb, img_dep, cameraInMatrix, mask)
    # 转到世界坐标系下
    scene_pcd = tf.transPts_T(scene_pcd, T_f2_f1=cameraPoseMatrix)
    # 去除工作范围外的点云
    scene_pcd_norm = scene_pcd - np.array([0, 0, 0.8])
    scene_pcd = np.delete(scene_pcd, np.where(np.abs(scene_pcd_norm) > 0.6)[0], axis=0)
    # FPS
    scene_pcd = tf.farthest_point_sample(scene_pcd, npoint=1024)
    # 删除离群点
    # scene_pcd = tf.removeOutLier_pcl(scene_pcd, nb_points=20, radius=0.1)
    # 补全
    # short_points_num = max(1024-scene_pcd.shape[0], 0)
    # if short_points_num > 0:
    #     extra_points = np.expand_dims(scene_pcd[0], axis=0).repeat(short_points_num, axis=0)
    #     scene_pcd = np.concatenate((scene_pcd, extra_points), axis=0)

    if visual:
        # show
        show_pcd(scene_pcd)
    
    # **************** object_pcd ****************
    object_pcd = env.get_object_pcd(num=1024)
    if visual:
        # show
        show_pcd(object_pcd)
        # show
        object_pcd_in_scene = tf.transPts_tq(object_pcd, object_goal_pose[:3], object_goal_pose[3:])
        pcd = np.concatenate((scene_pcd, object_pcd_in_scene), axis=0)
        show_pcd(pcd)
    
    if dtype is not None:
        scene_pcd = scene_pcd.astype(dtype)
        object_pcd = object_pcd.astype(dtype)
        object_goal_pose = object_goal_pose.astype(dtype)

    return scene_pcd, object_pcd, object_goal_pose


def get_scene_object_pcd_goal_toolhang(dataset_path, visual=False, n=1024, dtype=np.float32):
    """
    获取toolhang任务的scene_pcd / object_pcd / object_goal_pose
    物体点云由深度图得到，得到两个物体点云
    dataset_path: robomimic(toolhang)数据集路径
    visual: 是否可视化图像/pcd
    return:
        - scene_pcd: (N, 3)
        - frame_pcd: (N, 3)
        - tool_pcd: (N, 3)
        - object_goal_pose: (7,) pos+quat
    """
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map
    import time
    import hiera_diffusion_policy.common.transformation as tf

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    options = {}

    assert env_meta['env_name'] == 'ToolHang'
    options["env_name"] = env_meta['env_name']
    options["robots"] = env_meta['env_kwargs']["robots"]
    controller_name = "OSC_POSE"
    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)
    
    # ****************  ****************
    # initialize the task
    camera = ["toolhang_agentview", "toolhang_birdview"]
    segmentation_level = ['instance', 'instance']  # Options are {instance, class, element}
    camera_heights = 512
    camera_widths = 512
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_names=camera,
        camera_segmentations=segmentation_level,
        camera_depths=True,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
    )
    env.reset()
    
    # **************** object goal pose ****************
    object_goal_pose = env.object_goal_pose()

    # **************** bird image ****************
    for i in range(20):
        action = [-1, 0, 0, 0, 0, 0, 0]
        obs_bird, reward, done, _ = env.step(action)
    # segmentation
    # stand:1, frame:2, tool:3
    img_seg_bird = obs_bird[f"{camera[1]}_segmentation_{segmentation_level[1]}"].squeeze(-1)[::-1]
    img_seg_bird_objs = np.zeros_like(img_seg_bird, dtype=int)  
    img_seg_bird_objs[np.where(img_seg_bird == 2)] = 1  # frame
    img_seg_bird_objs[np.where(img_seg_bird == 3)] = 2  # tool
    # rgb
    img_rgb_bird = obs_bird[f"{camera[1]}_image"][::-1]
    # depth
    img_dep_bird = obs_bird[f"{camera[1]}_depth"].squeeze(-1)[::-1]
    img_dep_bird = get_real_depth_map(env.sim, img_dep_bird)

    # **************** front image ****************
    env.remove_all_objects()    # 移除所有物体
    action = [-1, 0, 0, 0, 0, 0, 0]
    obs_agent, reward, done, _ = env.step(action)
    # segmentation
    img_seg_agent = obs_agent[f"{camera[0]}_segmentation_{segmentation_level[0]}"].squeeze(-1)[::-1]
    img_seg_agent[np.where(img_seg_agent < 2)] = 0
    img_seg_agent[np.where(img_seg_agent > 0)] = 1
    # rgb
    img_rgb_agent = obs_agent[f"{camera[0]}_image"][::-1]
    # depth
    img_dep_agent = obs_agent[f"{camera[0]}_depth"].squeeze(-1)[::-1]
    img_dep_agent = get_real_depth_map(env.sim, img_dep_agent)

    if visual:
        show_rgb_seg_dep(camera[1], img_rgb_bird, img_seg_bird_objs, img_dep_bird)
        show_rgb_seg_dep(camera[0], img_rgb_agent, img_seg_agent, img_dep_agent)

    # **************** camera info ****************
    cameraInMatrix_agent = get_camera_intrinsic_matrix(env.sim, camera[0], camera_heights, camera_widths)
    cameraPoseMatrix_agent = get_camera_extrinsic_matrix(env.sim, camera[0])
    cameraInMatrix_bird = get_camera_intrinsic_matrix(env.sim, camera[1], camera_heights, camera_widths)
    cameraPoseMatrix_bird = get_camera_extrinsic_matrix(env.sim, camera[1])

    # **************** scene_pcd ****************
    mask = np.zeros(img_seg_agent.shape[:2], dtype=np.bool)
    mask[np.where(img_seg_agent == 0)] = 1
    scene_pcd = tf.create_point_cloud(img_rgb_agent, img_dep_agent, cameraInMatrix_agent, mask)
    # 转到世界坐标系下
    scene_pcd = tf.transPts_T(scene_pcd, T_f2_f1=cameraPoseMatrix_agent)
    # 去除工作范围外的点云
    scene_pcd_norm = scene_pcd - np.array([0, 0, 0.8])
    scene_pcd = np.delete(scene_pcd, np.where(np.abs(scene_pcd_norm) > 0.6)[0], axis=0)
    # FPS
    scene_pcd = tf.farthest_point_sample(scene_pcd, npoint=n)
    # 删除离群点
    # scene_pcd = tf.removeOutLier_pcl(scene_pcd, nb_points=20, radius=0.1)
    # 补全
    # short_points_num = max(1024-scene_pcd.shape[0], 0)
    # if short_points_num > 0:
    #     extra_points = np.expand_dims(scene_pcd[0], axis=0).repeat(short_points_num, axis=0)
    #     scene_pcd = np.concatenate((scene_pcd, extra_points), axis=0)

    if visual:
        show_pcd(scene_pcd)
    
    # **************** object_pcd (frame) ****************
    mask_frame = np.zeros(img_seg_bird_objs.shape[:2], dtype=np.bool)
    mask_frame[np.where(img_seg_bird_objs == 1)] = 1
    frame_pcd_camera = tf.create_point_cloud(img_rgb_bird, img_dep_bird, cameraInMatrix_bird, mask_frame)
    # 转到世界坐标系下
    frame_pcd_W = tf.transPts_T(frame_pcd_camera, T_f2_f1=cameraPoseMatrix_bird)
    # 删除离群点
    frame_pcd_W = tf.removeOutLier_pcl(frame_pcd_W, nb_points=30, radius=0.01)
    # 转到物体坐标系下
    T_W_frame = tf.PosQua_to_TransMat(obs_bird['frame_pos'], obs_bird['frame_quat'])
    frame_pcd = tf.transPts_T(frame_pcd_W, T_f2_f1=np.linalg.inv(T_W_frame))
    if frame_pcd.shape[0] > n:
        # FPS
        frame_pcd = tf.farthest_point_sample(frame_pcd, npoint=n)
    elif frame_pcd.shape[0] < n:
        # 补全
        short_points_num = n-frame_pcd.shape[0]
        extra_points = np.expand_dims(frame_pcd[0], axis=0).repeat(short_points_num, axis=0)
        frame_pcd = np.concatenate((frame_pcd, extra_points), axis=0)

    # **************** object_pcd (tool) ****************
    mask_tool = np.zeros(img_seg_bird_objs.shape[:2], dtype=np.bool)
    mask_tool[np.where(img_seg_bird_objs == 2)] = 1
    tool_pcd_camera = tf.create_point_cloud(img_rgb_bird, img_dep_bird, cameraInMatrix_bird, mask_tool)
    # 转到世界坐标系下
    tool_pcd_W = tf.transPts_T(tool_pcd_camera, T_f2_f1=cameraPoseMatrix_bird)
    # 删除离群点
    # tool_pcd_W = tf.removeOutLier_pcl(tool_pcd_W, nb_points=30, radius=0.01)
    # 转到物体坐标系下
    T_W_tool = tf.PosQua_to_TransMat(obs_bird['tool_pos'], obs_bird['tool_quat'])
    tool_pcd = tf.transPts_T(tool_pcd_W, T_f2_f1=np.linalg.inv(T_W_tool))
    if tool_pcd.shape[0] > n:
        # FPS
        tool_pcd = tf.farthest_point_sample(tool_pcd, npoint=n)
    elif tool_pcd.shape[0] < n:
        # 补全
        short_points_num = n-tool_pcd.shape[0]
        extra_points = np.expand_dims(tool_pcd[0], axis=0).repeat(short_points_num, axis=0)
        tool_pcd = np.concatenate((tool_pcd, extra_points), axis=0)

    if visual:
        # show objects pcd  in obj frames
        show_pcd(frame_pcd)
        show_pcd(tool_pcd_W)
    
    if dtype is not None:
        scene_pcd = scene_pcd.astype(dtype)
        frame_pcd = frame_pcd.astype(dtype)
        tool_pcd = tool_pcd.astype(dtype)
        object_goal_pose = object_goal_pose.astype(dtype)

    return scene_pcd, frame_pcd, tool_pcd, object_goal_pose


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

import numpy as np
from typing import Union
import open3d as o3d
import cv2
import matplotlib.cm as cm
import colorsys
import hiera_diffusion_policy.common.transformation as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import trimesh
import scipy
import time
import skimage.transform as st


def create_cube_pcl(size, pos, quat, num, usedForNormal=False):
    """
    生成立方体表面的点云

    args:
        - size: (3,) xyz方向的半尺寸，原点在cube中心
        - pos: (3,)  cube坐标系在参考坐标系中的位置
        - quat: (4,)  cube坐标系在参考坐标系中的四元数, xyzw
        - num: 点数量
    """
    l1, l2, l3 = size
    # 1/4在边缘上, 其他在表面上
    edge_radio = 0.5
    face_radio = 1-edge_radio
    len_edge = 2*l1*4 + 2*l2*4 + 2*l3*4
    num_edgeX = int((2*l1 / len_edge) * (num*edge_radio))
    num_edgeY = int((2*l2 / len_edge) * (num*edge_radio))
    num_edgeZ = int((2*l3 / len_edge) * (num*edge_radio))

    size_all = l1*l2*2 + l2*l3*2 + l1*l3*2
    num_up = int((l1*l2 / size_all) * (num*face_radio))
    num_left = int((l1*l3 / size_all) * (num*face_radio))
    num_forward = int((l2*l3 / size_all) * (num*face_radio))

    # 在每个边上增加点
    points_edge_x = np.random.rand(num_edgeX)*2*l1 - l1
    points_edge_x1 = np.ones((num_edgeX, 3)) * np.array([0, -l2, l3])
    points_edge_x2 = np.ones((num_edgeX, 3)) * np.array([0, l2, l3])
    points_edge_x3 = np.ones((num_edgeX, 3)) * np.array([0, l2, -l3])
    points_edge_x4 = np.ones((num_edgeX, 3)) * np.array([0, -l2, -l3])
    points_edge_x1[:, 0] = points_edge_x
    points_edge_x2[:, 0] = points_edge_x
    points_edge_x3[:, 0] = points_edge_x
    points_edge_x4[:, 0] = points_edge_x
    points_edge_xs = np.concatenate([points_edge_x1, points_edge_x2, points_edge_x3, points_edge_x4], axis=0)
    points_edge_y = np.random.rand(num_edgeY)*2*l2 - l2
    points_edge_y1 = np.ones((num_edgeY, 3)) * np.array([l1, 0, l3])
    points_edge_y2 = np.ones((num_edgeY, 3)) * np.array([-l1, 0, l3])
    points_edge_y3 = np.ones((num_edgeY, 3)) * np.array([-l1, 0, -l3])
    points_edge_y4 = np.ones((num_edgeY, 3)) * np.array([l1, 0, -l3])
    points_edge_y1[:, 1] = points_edge_y
    points_edge_y2[:, 1] = points_edge_y
    points_edge_y3[:, 1] = points_edge_y
    points_edge_y4[:, 1] = points_edge_y
    points_edge_ys = np.concatenate([points_edge_y1, points_edge_y2, points_edge_y3, points_edge_y4], axis=0)
    points_edge_z = np.random.rand(num_edgeZ)*2*l3 - l3
    points_edge_z1 = np.ones((num_edgeZ, 3)) * np.array([l1, l2, 0])
    points_edge_z2 = np.ones((num_edgeZ, 3)) * np.array([-l1, l2, 0])
    points_edge_z3 = np.ones((num_edgeZ, 3)) * np.array([-l1, -l2, 0])
    points_edge_z4 = np.ones((num_edgeZ, 3)) * np.array([l1, -l2, 0])
    points_edge_z1[:, 2] = points_edge_z
    points_edge_z2[:, 2] = points_edge_z
    points_edge_z3[:, 2] = points_edge_z
    points_edge_z4[:, 2] = points_edge_z
    points_edge_zs = np.concatenate([points_edge_z1, points_edge_z2, points_edge_z3, points_edge_z4], axis=0)
    points = np.concatenate([points_edge_xs, points_edge_ys, points_edge_zs], axis=0)
    num_edge = points.shape[0]
    
    # 在距离边缘0.016m以外的区域生成点云
    # 上表面 x: -l1 -> l1, y: -l2 -> l2, z: l3
    l = 0.008
    if usedForNormal:
        l1 -= l
        l2 -= l
    points_1 = np.random.rand(num_up, 3)
    points_1[:, 0] = points_1[:, 0]*2*l1 - l1
    points_1[:, 1] = points_1[:, 1]*2*l2 - l2
    points_1[:, 2] = points_1[:, 2] * 0 + l3
    # 下表面 x: -l1 -> l1, y: -l2 -> l2, z: l3
    points_2 = np.random.rand(num_up, 3)
    points_2[:, 0] = points_2[:, 0]*2*l1 - l1
    points_2[:, 1] = points_2[:, 1]*2*l2 - l2
    points_2[:, 2] = points_2[:, 2] * 0 - l3
    if usedForNormal:
        l1 += l
        l2 += l

        l1 -= l
        l3 -= l
    # 左表面 x: -l1 -> l1, y: -l2, z: -l3 -> l3
    points_3 = np.random.rand(num_left, 3)
    points_3[:, 0] = points_3[:, 0]*2*l1 - l1
    points_3[:, 1] = points_3[:, 1] * 0 - l2
    points_3[:, 2] = points_3[:, 2]*2*l3 - l3
    # 右表面 x: -l1 -> l1, y: l2, z: -l3 -> l3
    points_4 = np.random.rand(num_left, 3)
    points_4[:, 0] = points_4[:, 0]*2*l1 - l1
    points_4[:, 1] = points_4[:, 1] * 0 + l2
    points_4[:, 2] = points_4[:, 2]*2*l3 - l3
    if usedForNormal:
        l1 += l
        l3 += l

        l2 -= l
        l3 -= l
    # 前表面 x: l1, y: -l2 -> l2, z: -l3 -> l3
    points_5 = np.random.rand(num_forward, 3)
    points_5[:, 0] = points_5[:, 0] * 0 + l1
    points_5[:, 1] = points_5[:, 1]*2*l2 - l2
    points_5[:, 2] = points_5[:, 2]*2*l3 - l3
    # 后表面 x: -l1, y: -l2 -> l2, z: -l3 -> l3
    points_6 = np.random.rand(num_forward, 3)
    points_6[:, 0] = points_6[:, 0] * 0 - l1
    points_6[:, 1] = points_6[:, 1]*2*l2 - l2
    points_6[:, 2] = points_6[:, 2]*2*l3 - l3
    if usedForNormal:
        l2 += l
        l3 += l

    num_edge += points_2.shape[0]
    points = np.concatenate([points, points_2, points_1, points_3, points_4, points_5, points_6], axis=0)
    if points.shape[0] < num:
        points_append = np.zeros((num-points.shape[0], 3))
        for i in range(points_append.shape[0]):
            points_append[i, :] = points[-1, :]
        points = np.concatenate([points, points_append], axis=0)
    return tf.transPts_tq(points, pos, quat)

def calc_tbox_coverage(pose_target, pose_state):
    """
    用于真实moveT实验中计算coverage
    pose: (3,) xy坐标/z轴旋转
    """
    mask_target = create_tbox_mask(pose_target[:2], pose_target[2])
    mask_state = create_tbox_mask(pose_state[:2], pose_state[2])

    cv2.imshow('and', (mask_target & mask_state)*255)
    cv2.imshow('or', (mask_target | mask_state)*255)
    cv2.waitKey()

    total = np.sum(mask_target)
    i = np.sum(mask_target & mask_state)
    u = np.sum(mask_target | mask_state)
    iou = i / u
    coverage = i / total
    return coverage


def create_tbox_mask(pos, rot):
    """
    用于真实moveT实验
    构建tbox区域为1，其余位置为0的mask

    args:
        - pos: 二维位置
        - rot: z轴旋转弧度
    """
    # 先绘制原点下的tbox，再平移旋转


    # 左上角和右下角坐标(x,y), 单位m
    # 右侧bbox
    tbox1 = np.array([
        [-0.025, -0.05],
        [0.025, 0.1]
    ])
    # 左侧bbox
    tbox2 = np.array([
        [-0.1, -0.1],
        [0.1, -0.05]
    ])
    # 坐标翻转，加偏移，方便绘图
    tbox1 = tbox1[:, ::-1]*1000 + 500
    tbox1 = tbox1.astype(np.int64)
    tbox2 = tbox2[:, ::-1]*1000 + 500
    tbox2 = tbox2.astype(np.int64)

    img = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.rectangle(img, tbox1[0], tbox1[1], 1, -1)
    cv2.rectangle(img, tbox2[0], tbox2[1], 1, -1)

    # 旋转
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((500, 500), rot/np.pi*180., 1)
    img = cv2.warpAffine(img, M, (cols,rows))

    # 平移
    # 平移矩阵M：[[1,0,x],[0,1,y]]
    x_offset = int((pos[0]-0.5)*1000)
    y_offset = int(pos[1]*1000)
    M = np.float32([[1,0,y_offset],[0,1,x_offset]])
    img = cv2.warpAffine(img, M, (cols,rows))

    return img

def create_tbox_pts(num=700):
    """构建tbox物体的点云，尺寸为真实场景下movet任务中的尺寸
    return: (N, 3)
    """
    # 构建物体点云
    n1 = int(num*4./7.)
    n2 = num-n1
    o1 = create_cube_pcl((0.1, 0.025, 0.02), (0, -0.075, 0), (0, 0, 0, 1), n1)
    o2 = create_cube_pcl((0.025, 0.075, 0.02), (0, 0.025, 0), (0, 0, 0, 1), n2)
    pts = np.concatenate((o1, o2), axis=0)
    return pts

def create_pusht_pts(pts_num=1024):
    """构建T型物体的点云，尺寸为pusht任务中的尺寸
    return: (N, 2)
    """
    # 构建物体点云
    size_b = 120*30
    size_t = 90*30
    num_b = int(pts_num*size_b/(size_b+size_t))
    num_t = pts_num-num_b
    pts_b = np.random.rand(num_b, 2)
    pts_b[:, 0] = pts_b[:, 0]*120-60  # [0,1]->[-60,60]
    pts_b[:, 1] = pts_b[:, 1]*30  # [0,1]->[0,30]
    pts_t = np.random.rand(num_t, 2)
    pts_t[:, 0] = pts_t[:, 0]*30-15  # [0,1]->[-15,15]
    pts_t[:, 1] = pts_t[:, 1]*90+30  # [0,1]->[30,120]
    pts = np.concatenate([pts_b, pts_t], axis=0)
    return pts

def visual_pushT_dataset(replay_buffer, start_id=0, interv=3):
    """可视化pusht数据集"""
        
    # 构建物体点云
    pts = create_pusht_pts(pts_num=1024)
    
    for i in range(replay_buffer['keypoint'].shape[0])[start_id::interv]:
        print('='*10, i, '='*10)
        # print('state =', replay_buffer['state'][i])
        # print('action =', replay_buffer['action'][i])
        print('subgoal =', replay_buffer['subgoal'][i])
        print('reward =', replay_buffer['reward'][i])

        # 物体点云
        pos = replay_buffer['state'][i, 2:4]
        rot = replay_buffer['state'][i, 4]
        tf_img_obj = st.AffineTransform(translation=pos, rotation=rot)
        pts_global = tf_img_obj(pts)
        # 手指位置
        fin = Circle(
            (replay_buffer['state'][i, 0], replay_buffer['state'][i, 1]), 15, 
            color='black')
        # 子目标
        if replay_buffer['subgoal'][i, -1] > 0:
            fin_goal = Circle(
                (replay_buffer['subgoal'][i, 0], replay_buffer['subgoal'][i, 1]), 15, 
                color='r')
        # 关键点位置
        # kps_x = replay_buffer['keypoint'][i, :, 0]
        # kps_y = replay_buffer['keypoint'][i, :, 1]

        # 可视化
        fig, ax = plt.subplots()
        ax.scatter(pts_global[:, 0], pts_global[:, 1], c='b')
        # ax.scatter(kps_x, kps_y, c='r')
        ax.add_patch(fin)
        if replay_buffer['subgoal'][i, -1] > 0:
            ax.add_patch(fin_goal)
        ax.set_xlim([0,500])
        ax.set_ylim([0,500])
        ax.set_aspect('equal')
        plt.show()


def visual_pushT_pred_subgoals(state, subgoal):
    """
    state: (B, 5)
    subgoal: (B, 3)
    """
    obj_pcd = create_pusht_pts()
    for step in range(state.shape[0])[::5]:
        print('subgoal =', subgoal[step])
        # 物体点云
        pos = state[step, 2:4]
        rot = state[step, 4]
        tf_img_obj = st.AffineTransform(translation=pos, rotation=rot)
        pts_global = tf_img_obj(obj_pcd)
        # 手指位置
        fin = Circle((state[step, 0], state[step, 1]), 15, color='black')
        # 目标位置
        if subgoal[step, -1] > 0:
            fin_goal = Circle((subgoal[step, 0], subgoal[step, 1]), 15, color='r')

        # 可视化
        fig, ax = plt.subplots()
        ax.scatter(pts_global[:, 0], pts_global[:, 1], c='b')
        ax.add_patch(fin)
        if subgoal[step, -1] > 0:
            ax.add_patch(fin_goal)
        ax.set_xlim([0,500])
        ax.set_ylim([0,500])
        ax.set_aspect('equal')
        plt.show()

def visual_pushT_pred_subgoal(state, subgoal=None):
    """
    state: (5,)
    subgoal: (3,)
    """
    obj_pcd = create_pusht_pts()
    # 物体点云
    pos = state[2:4]
    rot = state[4]
    tf_img_obj = st.AffineTransform(translation=pos, rotation=rot)
    pts_global = tf_img_obj(obj_pcd)
    # 手指位置
    fin = Circle((state[0], state[1]), 15, color='black')
    # 目标位置
    if subgoal is not None and subgoal[-1] > 0:
        fin_goal = Circle((subgoal[0], subgoal[1]), 15, color='r')

    # 可视化
    fig, ax = plt.subplots()
    ax.scatter(pts_global[:, 0], pts_global[:, 1], c='b')
    ax.add_patch(fin)
    if subgoal is not None and subgoal[-1] > 0:
        ax.add_patch(fin_goal)
    ax.set_xlim([0,500])
    ax.set_ylim([0,500])
    ax.set_aspect('equal')
    plt.show()

def randomize_colors(N, bright=True):
    """
    Modified from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py#L59
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.5
    hsv = [(1.0 * i / N, 1, brightness) for i in range(N)]
    colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    rstate = np.random.RandomState(seed=20)
    np.random.shuffle(colors)
    return colors

def segmentation_to_rgb(seg_im, random_colors=False):
    """
    Helper function to visualize segmentations as RGB frames.
    NOTE: assumes that geom IDs go up to 255 at most - if not,
    multiple geoms might be assigned to the same color.
    """
    # ensure all values lie within [0, 255]
    seg_im = np.mod(seg_im, 256)

    if random_colors:
        colors = randomize_colors(N=256, bright=True)
        return (255.0 * colors[seg_im]).astype(np.uint8)
    else:
        # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
        rstate = np.random.RandomState(seed=8)
        inds = np.arange(256)
        rstate.shuffle(inds)

        # use @inds to map each geom ID to a color
        return (255.0 * cm.rainbow(inds[seg_im], 3)).astype(np.uint8)[..., :3]
    

def draw_pcl(pcl, colors=None):
    """
    可视化point cloud
    args:
        - pcl: (n, 3) points location
        - color: (n, 3) rgb颜色
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="pcd")


def depth2Gray(im_depth):
    """
    将深度图转至8位灰度图
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    # if x_max == x_min:
    #     print('图像渲染出错 ...')
    #     raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max
    return (im_depth * k + b).astype(np.uint8)

def depth2Gray3(im_depth):
    """
    将深度图转至三通道8位灰度图
    (h, w, 3)
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max

    ret = (im_depth * k + b).astype(np.uint8)
    ret = np.expand_dims(ret, 2).repeat(3, axis=2)
    return ret

def depth2RGB(im_depth):
    """
    将深度图转至三通道8位彩色图
    先将值为0的点去除，然后转换为彩图，然后将值为0的点设为红色
    (h, w, 3)
    im_depth: 单位 mm或m
    """
    im_depth = depth2Gray(im_depth)
    im_color = cv2.applyColorMap(im_depth, cv2.COLORMAP_JET)
    return im_color

def visual_obs_subgoal_matplot(
        raw_obs: dict, 
        obj_sub_goals: np.ndarray, 
        finger_sub_goals: np.ndarray,
        object_pcd: np.ndarray,
        finger_pcd: np.ndarray=None,):
    """
    visual current and subgoal pose/position of object and finger

    args:
        - raw_obs: h5py dict {
            - object  
            - robot0_eef_pos
            - robot0_eef_quat
            - robot0_gripper_qpos
        }
        obs['object'] start with object pose, shape=(N, S) N为当前轨迹长度，S为state维度
        - obj_sub_goals: object pose subgoals, np.adarray, shape=(N, 7) pos+quat
        - finger_sub_goals: finger position subgoals, np.adarray, shape=(N, 6), second dim denotes left and right fingers
        - object_pcd: object pointcloud, shape=(n, 3)
        - finger_pcd: finger pointcloud, shape=(n, 3), if not set, use default sphere pointcloud
    """
    if finger_pcd is None:
        finger_radius = 0.008
        ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
        finger_pcd = np.asarray(ft_mesh.vertices)

    trajectory_length = raw_obs['object'].shape[0]
    for step in range(trajectory_length):
        # current object pcd
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        current_object_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)

        # subgoal object pcd
        obj_pos_subgoal = obj_sub_goals[step, :3]
        obj_qua_subgoal = obj_sub_goals[step, 3:]
        subgoal_object_pcd = tf.transPts_tq(object_pcd, obj_pos_subgoal, obj_qua_subgoal)

        # current finger pcd
        gripper_width = (raw_obs['robot0_gripper_qpos'][step, 0]) * 2
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0],
            raw_obs['robot0_gripper_qpos'][step, 1],
            )
        current_fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        current_fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])

        # subgoal finger pcd
        fl_pos_subgoal = finger_sub_goals[step, :3]
        fr_pos_subgoal = finger_sub_goals[step, 3:]
        subgoal_fl_pcd = tf.transPts_tq(finger_pcd, fl_pos_subgoal, [0, 0, 0, 1])
        subgoal_fr_pcd = tf.transPts_tq(finger_pcd, fr_pos_subgoal, [0, 0, 0, 1])

        # show
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        current_object_pcd_color = np.array([[0, 0, 0]]).repeat(subgoal_object_pcd.shape[0], axis=0)/255.
        subgoal_object_pcd_color = np.array([[54, 54, 54]]).repeat(subgoal_object_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_object_pcd.transpose(1, 0)), color=current_object_pcd_color)
        ax.scatter(*tuple(subgoal_object_pcd.transpose(1, 0)), color=subgoal_object_pcd_color)

        current_fl_pcd_color = np.array([[255, 0, 0]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        subgoal_fl_pcd_color = np.array([[255, 140, 0]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_fl_pcd.transpose(1, 0)), color=current_fl_pcd_color)
        ax.scatter(*tuple(subgoal_fl_pcd.transpose(1, 0)), color=subgoal_fl_pcd_color)

        current_fr_pcd_color = np.array([[0, 0, 255]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        subgoal_fr_pcd_color = np.array([[0, 191, 255]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_fr_pcd.transpose(1, 0)), color=current_fr_pcd_color)
        ax.scatter(*tuple(subgoal_fr_pcd.transpose(1, 0)), color=subgoal_fr_pcd_color)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()


def visual_obs_relative_subgoal_matplot_robomimic(
        raw_obs: dict, 
        obj_sub_goals: np.ndarray, 
        finger_sub_goals: np.ndarray,
        object_pcd: np.ndarray,
        finger_pcd: np.ndarray=None,
        transform_init_pcd=False
        ):
    """
    visual current and subgoal pose/position of object and finger
    其中的物体子目标位姿是相对于物体的实时状态的新坐标系

    args:
        - raw_obs: h5py dict {
            - object  
            - robot0_eef_pos
            - robot0_eef_quat
            - robot0_gripper_qpos
        }
        obs['object'] start with object pose, shape=(N, S) N为当前轨迹长度，S为state维度
        - obj_sub_goals: object pose subgoals, np.adarray, shape=(N, 7) pos+quat
        - finger_sub_goals: finger position subgoals, np.adarray, shape=(N, 6), second dim denotes left and right fingers
        - object_pcd: object pointcloud, shape=(n, 3)
        - finger_pcd: finger pointcloud, shape=(n, 3), if not set, use default sphere pointcloud
    """
    if finger_pcd is None:
        finger_radius = 0.008
        ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
        finger_pcd = np.asarray(ft_mesh.vertices)

    trajectory_length = raw_obs['object'].shape[0]
    
    for step in range(trajectory_length):
        # current object pcd
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        current_object_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)

        # subgoal object pcd
        if transform_init_pcd:
            # 将 初始点云 转换到 子目标位姿
            # 首先计算 物体子目标新坐标系 相对于 世界坐标系 的转换
            object_pcd_center = np.mean(object_pcd, axis=0) # (n, 3)
            t_W_Oc = tf.transPt(object_pcd_center, None, obj_pos, obj_qua)
            T_W_Oc = tf.PosEuler_to_TransMat(t_W_Oc, [0, 0, 0])
            T_Oc_Oss = tf.PosQua_to_TransMat(obj_sub_goals[step, :3], obj_sub_goals[step, 3:])
            T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
            T_Oc_O = np.matmul(np.linalg.inv(T_W_Oc), T_W_O)
            T_W_Os = np.matmul(np.matmul(T_W_Oc, T_Oc_Oss), T_Oc_O)
            obj_pos_subgoal, obj_qua_subgoal = tf.TransMat_to_PosQua(T_W_Os)
            subgoal_object_pcd = tf.transPts_tq(object_pcd, obj_pos_subgoal, obj_qua_subgoal)
        else:
            # 将当前时刻的点云 转换到 子目标位姿
            # 当前时刻点云: current_object_pcd
            object_pcd_center = np.mean(current_object_pcd, axis=0) # (3)
            T_W_Oc = tf.PosQua_to_TransMat(object_pcd_center, (0, 0, 0, 1))
            init_pcd = current_object_pcd - object_pcd_center   # (n, 3) - (3,) = (n, 3)
            T_Oc_Oss = tf.PosQua_to_TransMat(obj_sub_goals[step, :3], obj_sub_goals[step, 3:])
            T_W_Oss = np.matmul(T_W_Oc, T_Oc_Oss)
            subgoal_object_pcd = tf.transPts_T(init_pcd, T_W_Oss)

        # current finger pcd
        gripper_width = (raw_obs['robot0_gripper_qpos'][step, 0]) * 2
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0],
            raw_obs['robot0_gripper_qpos'][step, 1]
            )
        current_fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        current_fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])

        # subgoal finger pcd
        fl_pos_subgoal = finger_sub_goals[step, :3]
        fr_pos_subgoal = finger_sub_goals[step, 3:]
        subgoal_fl_pcd = tf.transPts_tq(finger_pcd, fl_pos_subgoal, [0, 0, 0, 1])
        subgoal_fr_pcd = tf.transPts_tq(finger_pcd, fr_pos_subgoal, [0, 0, 0, 1])

        # show
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        current_object_pcd_color = np.array([[0, 0, 0]]).repeat(subgoal_object_pcd.shape[0], axis=0)/255.
        subgoal_object_pcd_color = np.array([[54, 54, 54]]).repeat(subgoal_object_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_object_pcd.transpose(1, 0)), color=current_object_pcd_color)
        ax.scatter(*tuple(subgoal_object_pcd.transpose(1, 0)), color=subgoal_object_pcd_color)

        current_fl_pcd_color = np.array([[255, 0, 0]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        subgoal_fl_pcd_color = np.array([[255, 140, 0]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_fl_pcd.transpose(1, 0)), color=current_fl_pcd_color)
        ax.scatter(*tuple(subgoal_fl_pcd.transpose(1, 0)), color=subgoal_fl_pcd_color)

        current_fr_pcd_color = np.array([[0, 0, 255]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        subgoal_fr_pcd_color = np.array([[0, 191, 255]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_fr_pcd.transpose(1, 0)), color=current_fr_pcd_color)
        ax.scatter(*tuple(subgoal_fr_pcd.transpose(1, 0)), color=subgoal_fr_pcd_color)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()

    
def visual_obs_relative_subgoal_matplot_nonprehensile(
        raw_obs: dict, 
        obj_sub_goals: np.ndarray, 
        finger_sub_goals: np.ndarray,
        object_pcd: np.ndarray,
        finger_pcd: np.ndarray=None,
        transform_init_pcd=False
        ):
    """
    visual current and subgoal pose/position of object and finger
    其中的物体子目标位姿是相对于物体的实时状态的新坐标系
    * 相比于 visual_obs_relative_subgoal_matplot_robomimic() 修改obs部分代码

    args:
        - raw_obs: h5py dict {
            - object_pos
            - object_quat
            - eef_pos
            - eef_quat
            - fingers_position
        }
        obs['object'] start with object pose, shape=(N, S) N为当前轨迹长度，S为state维度
        - obj_sub_goals: object pose subgoals, np.adarray, shape=(N, 7) pos+quat
        - finger_sub_goals: finger position subgoals, np.adarray, shape=(N, 6), second dim denotes left and right fingers
        - object_pcd: object pointcloud, shape=(n, 3)
        - finger_pcd: finger pointcloud, shape=(n, 3), if not set, use default sphere pointcloud
    """
    if finger_pcd is None:
        finger_radius = 0.008
        ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
        finger_pcd = np.asarray(ft_mesh.vertices)

    trajectory_length = raw_obs['object_pos'].shape[0]
    
    for step in range(trajectory_length):
        # current object pcd
        obj_pos = raw_obs['object_pos'][step]
        obj_qua = raw_obs['object_quat'][step]
        current_object_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)

        # subgoal object pcd
        if transform_init_pcd:
            # 将 初始点云 转换到 子目标位姿
            # 首先计算 物体子目标新坐标系 相对于 世界坐标系 的转换
            object_pcd_center = np.mean(object_pcd, axis=0) # (n, 3)
            t_W_Oc = tf.transPt(object_pcd_center, None, obj_pos, obj_qua)
            T_W_Oc = tf.PosEuler_to_TransMat(t_W_Oc, [0, 0, 0])
            T_Oc_Oss = tf.PosQua_to_TransMat(obj_sub_goals[step, :3], obj_sub_goals[step, 3:])
            T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
            T_Oc_O = np.matmul(np.linalg.inv(T_W_Oc), T_W_O)
            T_W_Os = np.matmul(np.matmul(T_W_Oc, T_Oc_Oss), T_Oc_O)
            obj_pos_subgoal, obj_qua_subgoal = tf.TransMat_to_PosQua(T_W_Os)
            subgoal_object_pcd = tf.transPts_tq(object_pcd, obj_pos_subgoal, obj_qua_subgoal)
        else:
            # 将当前时刻的点云 转换到 子目标位姿
            # 当前时刻点云: current_object_pcd
            object_pcd_center = np.mean(current_object_pcd, axis=0) # (3)
            T_W_Oc = tf.PosQua_to_TransMat(object_pcd_center, (0, 0, 0, 1))
            init_pcd = current_object_pcd - object_pcd_center   # (n, 3) - (3,) = (n, 3)
            T_Oc_Oss = tf.PosQua_to_TransMat(obj_sub_goals[step, :3], obj_sub_goals[step, 3:])
            T_W_Oss = np.matmul(T_W_Oc, T_Oc_Oss)
            subgoal_object_pcd = tf.transPts_T(init_pcd, T_W_Oss)

        # current finger pcd
        fl_pos = raw_obs['fingers_position'][step, :3]
        fr_pos = raw_obs['fingers_position'][step, 3:]
        current_fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        current_fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])

        # subgoal finger pcd
        fl_pos_subgoal = finger_sub_goals[step, :3]
        fr_pos_subgoal = finger_sub_goals[step, 3:]
        subgoal_fl_pcd = tf.transPts_tq(finger_pcd, fl_pos_subgoal, [0, 0, 0, 1])
        subgoal_fr_pcd = tf.transPts_tq(finger_pcd, fr_pos_subgoal, [0, 0, 0, 1])

        # show
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        current_object_pcd_color = np.array([[0, 0, 0]]).repeat(subgoal_object_pcd.shape[0], axis=0)/255.
        subgoal_object_pcd_color = np.array([[54, 54, 54]]).repeat(subgoal_object_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_object_pcd.transpose(1, 0)), color=current_object_pcd_color)
        ax.scatter(*tuple(subgoal_object_pcd.transpose(1, 0)), color=subgoal_object_pcd_color)

        current_fl_pcd_color = np.array([[255, 0, 0]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        subgoal_fl_pcd_color = np.array([[255, 140, 0]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_fl_pcd.transpose(1, 0)), color=current_fl_pcd_color)
        ax.scatter(*tuple(subgoal_fl_pcd.transpose(1, 0)), color=subgoal_fl_pcd_color)

        current_fr_pcd_color = np.array([[0, 0, 255]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        subgoal_fr_pcd_color = np.array([[0, 191, 255]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_fr_pcd.transpose(1, 0)), color=current_fr_pcd_color)
        ax.scatter(*tuple(subgoal_fr_pcd.transpose(1, 0)), color=subgoal_fr_pcd_color)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()


def visual_subgoals_tilt_v41(state, reward, subgoals, scene_pcd, object_pcd):
    """
    可视化subgoal
    args:
        - state: (N, S) 物体位姿7/机械臂末端位姿7/两个手指的位置6
        - subgoals: {obj_subgoals; fin_subgoals}
    """
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    for step in range(state.shape[0]):
        print('step:', step, 'reward:', reward[step])
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        # visual scene
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # visual current object
        obj_pos = state[step, :3]
        obj_qua = state[step, 3:7]
        current_obj_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)
        obj_color = np.array([[139, 105, 20]]).repeat(current_obj_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_obj_pcd.transpose(1, 0)), color=obj_color)

        # visual current finger
        fl_pos = state[step, -6:-3]
        fr_pos = state[step, -3:]
        fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])
        fl_pcd_color = np.array([[0, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
        fr_pcd_color = np.array([[0, 0, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_pcd_color)
        ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_pcd_color)

        # visual obj subgoals
        obj_subgoals = subgoals['obj_subgoal'] # (n, 7)
        fin_subgoals = subgoals['fin_subgoal'] # (n, 6)
        sg_num = obj_subgoals.shape[0]
        obj_sg_colors = Color.gradient_colors(sg_num, start_c=[0, 0, 255], end_c=[0, 191, 255]) # 蓝
        fl_sg_colors = Color.gradient_colors(sg_num, start_c=[255, 0, 0], end_c=[255, 0, 255])  # 红
        fr_sg_colors = Color.gradient_colors(sg_num, start_c=[34, 139, 34], end_c=[124, 252, 0])  # 绿
        for i in range(sg_num):
            obj_pcd = tf.transPts_tq(object_pcd, obj_subgoals[i, :3], obj_subgoals[i, 3:])
            fl_pcd = tf.transPts_tq(finger_pcd, fin_subgoals[i, :3], (0, 0, 0, 1))
            fr_pcd = tf.transPts_tq(finger_pcd, fin_subgoals[i, 3:], (0, 0, 0, 1))
            ax.scatter(*tuple(obj_pcd.transpose(1, 0)), color=obj_sg_colors[i])
            ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_sg_colors[i])
            ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_sg_colors[i])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()


def visual_subgoals_tilt_v43_1(state, reward, subgoals, scene_pcd, object_pcd):
    """
    可视化subgoal
    args:
        - state: (N, S) 物体位姿7/机械臂末端位姿7/两个手指的位置6
        - subgoals: (n, 7+8)
    """
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    for step in range(state.shape[0]):
        print('step:', step, 'reward:', reward[step])
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        # visual scene
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # visual current object
        obj_pos = state[step, :3]
        obj_qua = state[step, 3:7]
        current_obj_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)
        obj_color = np.array([[139, 105, 20]]).repeat(current_obj_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_obj_pcd.transpose(1, 0)), color=obj_color)

        # visual current finger
        fl_pos = state[step, -6:-3]
        fr_pos = state[step, -3:]
        fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])
        fl_pcd_color = np.array([[0, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
        fr_pcd_color = np.array([[0, 0, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_pcd_color)
        ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_pcd_color)

        # visual obj subgoals
        obj_subgoals = subgoals[:, :7] # (n, 7)
        fin_subgoals = subgoals[:, 7:] # (n, 6)
        sg_num = obj_subgoals.shape[0]
        obj_sg_colors = Color.gradient_colors(sg_num, start_c=[0, 0, 255], end_c=[0, 191, 255]) # 蓝
        fl_sg_colors = Color.gradient_colors(sg_num, start_c=[255, 0, 0], end_c=[255, 0, 255])  # 红
        fr_sg_colors = Color.gradient_colors(sg_num, start_c=[34, 139, 34], end_c=[124, 252, 0])  # 绿
        for i in range(sg_num):
            obj_pcd = tf.transPts_tq(object_pcd, obj_subgoals[i, :3], obj_subgoals[i, 3:])
            fl_pcd = tf.transPts_tq(finger_pcd, fin_subgoals[i, :3], (0, 0, 0, 1))
            fr_pcd = tf.transPts_tq(finger_pcd, fin_subgoals[i, 3:6], (0, 0, 0, 1))
            ax.scatter(*tuple(obj_pcd.transpose(1, 0)), color=obj_sg_colors[i])
            ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_sg_colors[i])
            ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_sg_colors[i])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()

        break


def visual_subgoals_tilt_v43(state, subgoal_all, reward, scene_pcd, object_pcd):
    """
    可视化subgoal
    args:
        - state: (N, S) 物体位姿7/机械臂末端位姿7/两个手指的位置6
        - subgoals: (N, 7+8) 物体位姿/手指位置/是否接触
    """
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    for step in range(state.shape[0]):
        print('step:', step, 'reward:', reward[step])
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        # visual scene
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # visual current object
        obj_pos = state[step, :3]
        obj_qua = state[step, 3:7]
        current_obj_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)
        obj_color = np.array([[139, 105, 20]]).repeat(current_obj_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_obj_pcd.transpose(1, 0)), color=obj_color)

        # visual current finger
        fl_pos = state[step, -6:-3]
        fr_pos = state[step, -3:]
        fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])
        fl_pcd_color = np.array([[0, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
        fr_pcd_color = np.array([[0, 0, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_pcd_color)
        ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_pcd_color)

        # visual subgoals
        # object
        obj_sg_pcd = tf.transPts_tq(object_pcd, subgoal_all[step, :3], subgoal_all[step, 3:7])
        obj_sg_color = np.array([[0, 0, 255]]).repeat(obj_sg_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(obj_sg_pcd.transpose(1, 0)), color=obj_sg_color)
        # fl
        if subgoal_all[step, -2] == 1:
            fl_sg_pcd = tf.transPts_tq(finger_pcd, subgoal_all[step, 7:10], (0, 0, 0, 1))
            fl_sg_color = np.array([[255, 0, 0]]).repeat(fl_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fl_sg_pcd.transpose(1, 0)), color=fl_sg_color)
        # fr
        if subgoal_all[step, -1] == 1:
            fr_sg_pcd = tf.transPts_tq(finger_pcd, subgoal_all[step, 10:13], (0, 0, 0, 1))
            fr_sg_color = np.array([[34, 139, 34]]).repeat(fr_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fr_sg_pcd.transpose(1, 0)), color=fr_sg_color)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()


def visual_subgoals_tilt_v44(state, subgoal_all, reward, scene_pcd, object_pcd):
    """
    可视化subgoal
    args:
        - state: (N, S) 物体位姿7/机械臂末端位姿7/两个手指的位置6
        - subgoals: (N, 8) 手指位置/是否接触
    """
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    for step in range(state.shape[0])[::5]:
        print('step:', step, 'reward:', reward[step])
        print('subgoal =', subgoal_all[step])
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        # visual scene
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # visual current object
        obj_pos = state[step, :3]
        obj_qua = state[step, 3:7]
        current_obj_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)
        obj_color = np.array([[139, 105, 20]]).repeat(current_obj_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_obj_pcd.transpose(1, 0)), color=obj_color)

        # visual current finger
        fl_pos = state[step, -6:-3]
        fr_pos = state[step, -3:]
        fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])
        fl_pcd_color = np.array([[0, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
        fr_pcd_color = np.array([[0, 0, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_pcd_color)
        ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_pcd_color)

        # visual subgoals
        # fl
        if subgoal_all[step, 6] == 1:
            fl_sg_pcd = tf.transPts_tq(finger_pcd, subgoal_all[step, :3], (0, 0, 0, 1))
            fl_sg_color = np.array([[255, 0, 0]]).repeat(fl_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fl_sg_pcd.transpose(1, 0)), color=fl_sg_color)
        # fr
        if subgoal_all[step, 7] == 1:
            fr_sg_pcd = tf.transPts_tq(finger_pcd, subgoal_all[step, 3:6], (0, 0, 0, 1))
            fr_sg_color = np.array([[34, 139, 34]]).repeat(fr_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fr_sg_pcd.transpose(1, 0)), color=fr_sg_color)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()

def visual_subgoals_tilt_v44611(state, subgoal_all, subgoal_id, reward, scene_pcd, object_pcd):
    """
    可视化subgoal
    args:
        - state: (N, S) 物体位姿7/机械臂末端位姿7/两个手指的位置6
        - subgoals: (N, 8) 手指位置/是否接触
    """
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    for step in range(state.shape[0])[::3]:
        print('step:', step, 'reward:', reward[step])
        print('subgoal id =', subgoal_id[step])
        print('subgoal =', subgoal_all[step])
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        # visual scene
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # visual current object
        obj_pos = state[step, :3]
        obj_qua = state[step, 3:7]
        current_obj_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)
        obj_color = np.array([[139, 105, 20]]).repeat(current_obj_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_obj_pcd.transpose(1, 0)), color=obj_color)

        # visual current finger
        fl_pos = state[step, -6:-3]
        fr_pos = state[step, -3:]
        fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])
        fl_pcd_color = np.array([[0, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
        fr_pcd_color = np.array([[0, 0, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_pcd_color)
        ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_pcd_color)

        # visual subgoals
        # fl
        if subgoal_all[step, 6] == 1:
            fl_sg_pcd = tf.transPts_tq(finger_pcd, subgoal_all[step, :3], (0, 0, 0, 1))
            fl_sg_color = np.array([[255, 0, 0]]).repeat(fl_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fl_sg_pcd.transpose(1, 0)), color=fl_sg_color)
        # fr
        if subgoal_all[step, 7] == 1:
            fr_sg_pcd = tf.transPts_tq(finger_pcd, subgoal_all[step, 3:6], (0, 0, 0, 1))
            fr_sg_color = np.array([[34, 139, 34]]).repeat(fr_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fr_sg_pcd.transpose(1, 0)), color=fr_sg_color)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()

def visual_subgoals_v6(state, subgoal, reward, scene_pcd, object_pcd):
    """可视化subgoal"""
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    for step in range(state.shape[0])[::3]:
        # if sum(subgoal[step][-2:]) != 0:
        #     continue
        print('='*20)
        print('step:', step, 'reward:', reward[step])
        print('subgoal =', subgoal[step])
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        # visual scene
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # visual current object
        obj_pos = state[step, :3]
        obj_qua = state[step, 3:7]
        current_obj_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)
        obj_color = np.array([[139, 105, 20]]).repeat(current_obj_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_obj_pcd.transpose(1, 0)), color=obj_color)

        # visual current finger
        fl_pos = state[step, -6:-3]
        fr_pos = state[step, -3:]
        fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])
        fl_pcd_color = np.array([[0, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
        fr_pcd_color = np.array([[0, 0, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_pcd_color)
        ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_pcd_color)

        # visual subgoals
        # fl
        if subgoal[step, 6] == 1:
            fl_sg_pcd = tf.transPts_tq(finger_pcd, subgoal[step, :3], (0, 0, 0, 1))
            fl_sg_color = np.array([[255, 0, 0]]).repeat(fl_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fl_sg_pcd.transpose(1, 0)), color=fl_sg_color)
        # fr
        if subgoal[step, 7] == 1:
            fr_sg_pcd = tf.transPts_tq(finger_pcd, subgoal[step, 3:6], (0, 0, 0, 1))
            fr_sg_color = np.array([[34, 139, 34]]).repeat(fr_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fr_sg_pcd.transpose(1, 0)), color=fr_sg_color)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()

def visual_subgoals_real_v6(finger_pos, subgoal, reward, pcd, action=None):
    """可视化subgoal"""
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    for step in range(finger_pos.shape[0])[::10]:
        # if sum(subgoal[step][-2:]) != 0:
        #     continue
        print('='*20)
        print('step:', step, 'reward:', reward[step])
        print('subgoal =', subgoal[step])
        if action is not None:
            print('action =', action[step])

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        # visual scene
        if pcd.shape[-1] == 4:
            pcd_color = np.repeat(np.expand_dims(pcd[step, :, 3], axis=1), repeats=3, axis=1)
            ax.scatter(*tuple(pcd[step, :, :3].transpose(1, 0)), color=pcd_color)
        else:
            ax.scatter(*tuple(pcd[step, :, :3].transpose(1, 0)))

        # visual current finger
        fl_pos = finger_pos[step, :3]
        fr_pos = finger_pos[step, 3:]
        fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])
        fl_pcd_color = np.array([[0, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
        fr_pcd_color = np.array([[0, 0, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_pcd_color)
        ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_pcd_color)

        # visual subgoals
        # fl
        if subgoal[step, 6] == 1:
            fl_sg_pcd = tf.transPts_tq(finger_pcd, subgoal[step, :3], (0, 0, 0, 1))
            fl_sg_color = np.array([[255, 0, 0]]).repeat(fl_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fl_sg_pcd.transpose(1, 0)), color=fl_sg_color)
        # fr
        if subgoal[step, 7] == 1:
            fr_sg_pcd = tf.transPts_tq(finger_pcd, subgoal[step, 3:6], (0, 0, 0, 1))
            fr_sg_color = np.array([[34, 139, 34]]).repeat(fr_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fr_sg_pcd.transpose(1, 0)), color=fr_sg_color)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # ax.set_zlim(0.79, 1.05)
        # plt.xticks(np.arange(-0.3, 0.3, 0.05))
        # plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()
        
def visual_subgoals_pusht3d_v6(state, subgoal, reward, scene_pcd, object_pcd):
    """可视化subgoal
    state: (N, 10) obj_pose7/eef_pos3
    subgoal: (N, 4) pos3/val1
    """
    # build finger pcd
    finger_radius = 0.015
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    for step in range(state.shape[0])[::5]:
        print('='*20)
        print('step:', step, 'reward:', reward[step])
        print('subgoal =', subgoal[step])
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        # visual scene
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # visual current object
        obj_pos = state[step, :3]
        obj_qua = state[step, 3:7]
        current_obj_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)
        obj_color = np.array([[139, 105, 20]]).repeat(current_obj_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_obj_pcd.transpose(1, 0)), color=obj_color)

        # visual current finger
        f_pos = state[step, -3:]
        f_pcd = tf.transPts_tq(finger_pcd, f_pos, [0, 0, 0, 1])
        f_pcd_color = np.array([[0, 0, 0]]).repeat(f_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(f_pcd.transpose(1, 0)), color=f_pcd_color)

        # visual subgoals
        if subgoal[step, -1] == 1:
            fl_sg_pcd = tf.transPts_tq(finger_pcd, subgoal[step, :3], (0, 0, 0, 1))
            fl_sg_color = np.array([[255, 0, 0]]).repeat(fl_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fl_sg_pcd.transpose(1, 0)), color=fl_sg_color)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()


def show_rgb_seg_dep(camera_name, img_rgb, img_seg, img_dep):
    import matplotlib.pyplot as plt
    # from hiera_diffusion_policy.common.visual import depth2RGB

    plt.figure(figsize=(10,5)) #设置窗口大小
    plt.suptitle(camera_name) # 图片名称
    plt.subplot(2,2,1), plt.title('RGB')
    plt.imshow(img_rgb)
    plt.subplot(2,2,2), plt.title('Seg')
    plt.imshow(segmentation_to_rgb(img_seg))
    plt.subplot(2,2,3), plt.title('Dep')
    plt.imshow(depth2RGB(img_dep))
    plt.show()

def show_pcd(scene_pcd):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # scene
    ax.scatter(*tuple(scene_pcd.transpose(1, 0)))
    # show
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # plt.savefig('savefig_example.png')
    # plt.gca().set_box_aspect((2, 2, 1))  # 当x、y、z轴范围之比为3:5:2时。
    plt.show()

def visual_subgoals_v44(state, subgoal_all, subgoal_id, reward, scene_pcd, object_pcd):
    """
    可视化subgoal
    args:
        - state: (N, S) 物体位姿7/机械臂末端位姿7/两个手指的位置6
        - subgoals: (N, 8) 手指位置/是否接触
    """
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    for step in range(state.shape[0]):
        print('='*20)
        print('step:', step, 'reward:', reward[step])
        print('subgoal =', subgoal_all[step])
        print('subgoal_id =', subgoal_id[step])
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        # visual scene
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # visual current object
        obj_pos = state[step, :3]
        obj_qua = state[step, 3:7]
        current_obj_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)
        obj_color = np.array([[139, 105, 20]]).repeat(current_obj_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_obj_pcd.transpose(1, 0)), color=obj_color)

        # visual current finger
        fl_pos = state[step, -6:-3]
        fr_pos = state[step, -3:]
        fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])
        fl_pcd_color = np.array([[0, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
        fr_pcd_color = np.array([[0, 0, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_pcd_color)
        ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_pcd_color)

        # visual subgoals
        # fl
        if subgoal_all[step, 6] == 1:
            fl_sg_pcd = tf.transPts_tq(finger_pcd, subgoal_all[step, :3], (0, 0, 0, 1))
            fl_sg_color = np.array([[255, 0, 0]]).repeat(fl_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fl_sg_pcd.transpose(1, 0)), color=fl_sg_color)
        # fr
        if subgoal_all[step, 7] == 1:
            fr_sg_pcd = tf.transPts_tq(finger_pcd, subgoal_all[step, 3:6], (0, 0, 0, 1))
            fr_sg_color = np.array([[34, 139, 34]]).repeat(fr_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fr_sg_pcd.transpose(1, 0)), color=fr_sg_color)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()



def visual_subgoals_tilt_v44_1(init_state, subgoals, scene_pcd, object_pcd):
    """
    可视化subgoal
    args:
        - init_state: (S,) 物体位姿7/机械臂末端位姿7/两个手指的位置6
        - subgoals: (n, 8) 手指位置/是否接触
    """
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    sg_num = subgoals.shape[0]
    for i in range(sg_num):
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')
        
        # visual scene
        # ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # visual current object
        # obj_pos = init_state[:3]
        # obj_qua = init_state[3:7]
        # init_obj_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)
        # init_obj_color = np.array([[139, 105, 20]]).repeat(init_obj_pcd.shape[0], axis=0)/255.
        # ax.scatter(*tuple(init_obj_pcd.transpose(1, 0)), color=init_obj_color)

        # vis obj
        obj_color = np.array([[139, 105, 20]]).repeat(object_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(object_pcd.transpose(1, 0)), color=obj_color)

        # vis finger
        if subgoals[i, 6] == 1:
            fl_pcd = tf.transPts_tq(finger_pcd, subgoals[i, :3], (0, 0, 0, 1))
            fl_color = np.array([[255, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_color)

        if subgoals[i, 7] == 1:
            fr_pcd = tf.transPts_tq(finger_pcd, subgoals[i, 3:6], (0, 0, 0, 1))
            fr_color = np.array([[0, 255, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_color)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # ax.set_zlim(-0.1, 1.05)
        # plt.xticks(np.arange(-0.3, 0.3, 0.05))
        # plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()

def visual_subgoals_v44_1(init_state, subgoals, scene_pcd, object_pcd):
    """
    可视化subgoal
    args:
        - init_state: (S,) 物体位姿7/机械臂末端位姿7/两个手指的位置6
        - subgoals: (n, 8) 手指位置/是否接触
    """
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    sg_num = subgoals.shape[0]
    for i in range(sg_num):
        print('g =', i)
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')
        
        # visual scene
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # visual current object
        obj_pos = init_state[:3]
        obj_qua = init_state[3:7]
        init_obj_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)
        init_obj_color = np.array([[139, 105, 20]]).repeat(init_obj_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(init_obj_pcd.transpose(1, 0)), color=init_obj_color)

        # vis obj
        obj_color = np.array([[139, 105, 20]]).repeat(object_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(object_pcd.transpose(1, 0)), color=obj_color)

        # vis finger
        # if subgoals[i, 6] == 1:
        fl_pcd = tf.transPts_tq(finger_pcd, subgoals[i, :3], (0, 0, 0, 1))
        fl_color = np.array([[255, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_color)

        # if subgoals[i, 7] == 1:
        fr_pcd = tf.transPts_tq(finger_pcd, subgoals[i, 3:6], (0, 0, 0, 1))
        fr_color = np.array([[0, 255, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_color)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # ax.set_zlim(-0.1, 1.05)
        # plt.xticks(np.arange(-0.3, 0.3, 0.05))
        # plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()



def visual_subgoals_v446(init_obj_pcd, subgoals, scene_pcd):
    """
    可视化subgoal
    args:
        - init_state: (S,) 物体位姿7/机械臂末端位姿7/两个手指的位置6
        - subgoals: (n, 8) 手指位置/是否接触
    """
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    sg_num = subgoals.shape[0]
    for i in range(sg_num):
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')
        
        # visual scene
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # visual init object
        init_obj_color = np.array([[139, 105, 20]]).repeat(init_obj_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(init_obj_pcd.transpose(1, 0)), color=init_obj_color)

        # vis finger
        if subgoals[i, 6] == 1:
            fl_pcd = tf.transPts_tq(finger_pcd, subgoals[i, :3], (0, 0, 0, 1))
            fl_color = np.array([[255, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_color)

        if subgoals[i, 7] == 1:
            fr_pcd = tf.transPts_tq(finger_pcd, subgoals[i, 3:6], (0, 0, 0, 1))
            fr_color = np.array([[0, 255, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_color)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()


def visual_v4462(init_obj_pcd, cur_obj_pcd, scene_pcd):
    """
    可视化subgoal
    args:
        - init_state: (S,) 物体位姿7/机械臂末端位姿7/两个手指的位置6
        - subgoals: (n, 8) 手指位置/是否接触
    """

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    
    # visual scene
    ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

    # visual init object
    init_obj_color = np.array([[139, 105, 20]]).repeat(init_obj_pcd.shape[0], axis=0)/255.
    ax.scatter(*tuple(init_obj_pcd.transpose(1, 0)), color=init_obj_color)

    # visual cur object
    cue_obj_color = np.array([[0, 0, 255]]).repeat(cur_obj_pcd.shape[0], axis=0)/255.
    ax.scatter(*tuple(cur_obj_pcd.transpose(1, 0)), color=cue_obj_color)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def visual_subgoals_tilt_v44_2(state, subgoal=None, scene_pcd=None, object_pcd=None):
    """
    可视化subgoal
    args:
        - state: (S,) 物体位姿7/机械臂末端位姿7/两个手指的位置6
        - subgoal: (8, ) 手指位置/是否接触
    """
    # build finger pcd
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    
    # visual scene
    ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

    # visual current object
    obj_pos = state[:3]
    obj_qua = state[3:7]
    obj_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)
    obj_color = np.array([[139, 105, 20]]).repeat(obj_pcd.shape[0], axis=0)/255.
    ax.scatter(*tuple(obj_pcd.transpose(1, 0)), color=obj_color)

    # visual current finger
    fl_pcd = tf.transPts_tq(finger_pcd, state[-6:-3], (0, 0, 0, 1))
    ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=Color.color('black'))
    fr_pcd = tf.transPts_tq(finger_pcd, state[-3:], (0, 0, 0, 1))
    ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=Color.color('black'))

    if subgoal is not None:
        # vis finger
        if subgoal[6] == 1:
            fl_pcd = tf.transPts_tq(finger_pcd, subgoal[:3], (0, 0, 0, 1))
            fl_color = np.array([[255, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_color)

        if subgoal[7] == 1:
            fr_pcd = tf.transPts_tq(finger_pcd, subgoal[3:6], (0, 0, 0, 1))
            fr_color = np.array([[0, 255, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_color)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_zlim(0.7, 1.05)
    plt.xticks(np.arange(-0.4, 0.4, 0.05))
    plt.yticks(np.arange(-0.4, 0.4, 0.05))
    plt.show()


def visual_pcd(pcd):
    """
    可视化pcd
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    
    # visual scene
    ax.scatter(*tuple(pcd.transpose(1, 0)), color=Color.color('black'))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # ax.set_zlim(0.7, 1.05)
    # plt.xticks(np.arange(-0.4, 0.4, 0.05))
    # plt.yticks(np.arange(-0.4, 0.4, 0.05))
    plt.show()

def visual_subgoals_tilt(state, subgoals, scene_pcd, object_pcd, finger_pcd=None):
    """
    可视化subgoal
    args:
        - state: (N, S), 物体位姿7/机械臂末端位姿7/两个手指的位置6
        - subgoals: {obj_subgoal_world; obj_subgoal_obspcd; fin_subgoal_world}
        - object_pcd: object pointcloud, shape=(n, 3)
        - finger_pcd: finger pointcloud, shape=(n, 3), if not set, use default sphere pointcloud
    """
    if finger_pcd is None:
        finger_radius = 0.008
        ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
        finger_pcd = np.asarray(ft_mesh.vertices)

    obj_subgoal_world = subgoals['obj_subgoal_world']
    obj_subgoal_obspcd = subgoals['obj_subgoal_obspcd']
    fin_subgoal_world = subgoals['fin_subgoal_world']

    trajectory_length = state.shape[0]
    for step in range(trajectory_length):
        # current object pcd
        obj_pos = state[step, :3]
        obj_qua = state[step, 3:7]
        current_object_pcd = tf.transPts_tq(object_pcd, obj_pos, obj_qua)

        # subgoal object pcd
        object_pcd_center = np.mean(current_object_pcd, axis=0) # (3)
        obspcd = current_object_pcd - object_pcd_center   # (n, 3) - (3,) = (n, 3)
        T_W_Oc = tf.PosQua_to_TransMat(object_pcd_center, (0, 0, 0, 1))
        T_Oc_Oss = tf.PosQua_to_TransMat(obj_subgoal_obspcd[step, :3], obj_subgoal_obspcd[step, 3:])
        T_W_Oss = np.matmul(T_W_Oc, T_Oc_Oss)
        subgoal_object_pcd = tf.transPts_T(obspcd, T_W_Oss)

        # current finger pcd
        fl_pos = state[step, 14:17]
        fr_pos = state[step, 17:]
        current_fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        current_fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])

        # subgoal finger pcd
        fl_pos_subgoal = fin_subgoal_world[step, :3]
        fr_pos_subgoal = fin_subgoal_world[step, 3:6]
        fl_sg_contact = fin_subgoal_world[step, 6]
        fr_sg_contact = fin_subgoal_world[step, 7]
        subgoal_fl_pcd = tf.transPts_tq(finger_pcd, fl_pos_subgoal, [0, 0, 0, 1])
        subgoal_fr_pcd = tf.transPts_tq(finger_pcd, fr_pos_subgoal, [0, 0, 0, 1])

        print('fl_pos =', fl_pos)
        print('fr_pos =', fr_pos)
        print('fl_pos_subgoal =', fl_pos_subgoal)
        print('fr_pos_subgoal =', fr_pos_subgoal)

        # 计算手指位置与子目标的距离
        fl_diff = np.linalg.norm(fl_pos_subgoal - fl_pos)
        fr_diff = np.linalg.norm(fr_pos_subgoal - fr_pos)
        print('fl_diff =', fl_diff)
        print('fr_diff =', fr_diff)

        # show
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')
        # show object pcd
        # 观测黑色，子目标红色
        current_object_pcd_color = np.array([[0, 0, 0]]).repeat(subgoal_object_pcd.shape[0], axis=0)/255.
        subgoal_object_pcd_color = np.array([[255, 0, 0]]).repeat(subgoal_object_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_object_pcd.transpose(1, 0)), color=current_object_pcd_color)
        ax.scatter(*tuple(subgoal_object_pcd.transpose(1, 0)), color=subgoal_object_pcd_color)

        # show left finger
        # 观测绿色, 子目标 接触深蓝色，非接触浅蓝
        current_fl_pcd_color = np.array([[0, 255, 0]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        current_fr_pcd_color = np.array([[0, 255, 0]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(current_fl_pcd.transpose(1, 0)), color=current_fl_pcd_color)
        ax.scatter(*tuple(current_fr_pcd.transpose(1, 0)), color=current_fr_pcd_color)
        
        if fl_sg_contact:
            subgoal_fl_pcd_color = np.array([[0, 0, 205]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        else:
            subgoal_fl_pcd_color = np.array([[0, 191, 255]]).repeat(current_fl_pcd.shape[0], axis=0)/255.

        if fr_sg_contact:
            subgoal_fr_pcd_color = np.array([[0, 0, 205]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        else:
            subgoal_fr_pcd_color = np.array([[0, 191, 255]]).repeat(current_fl_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(subgoal_fl_pcd.transpose(1, 0)), color=subgoal_fl_pcd_color)
        ax.scatter(*tuple(subgoal_fr_pcd.transpose(1, 0)), color=subgoal_fr_pcd_color)
        
        # scene
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()


def visual_obj_subgoals_tilt(state, obj_subgoals, scene_pcd, object_pcd):
    """
    可视化物体subgoal
    args:
        - obj_subgoals: (N, 7) 物体子目标序列, 世界坐标系下
        - object_pcd: object pointcloud, shape=(n, 3)
    """
    print('子目标数量:', obj_subgoals.shape[0])
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    # scene
    ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))
    # 设置渐变颜色
    obj_colors = Color.gradient_colors(obj_subgoals.shape[0]+1, start_c=[255, 0, 0], end_c=[0, 0, 255])
    # 初始观测点云
    obj_pos_init = state[0, :3]
    obj_qua_init = state[0, 3:7]
    object_pcd_init = tf.transPts_tq(object_pcd, obj_pos_init, obj_qua_init)
    ax.scatter(*tuple(object_pcd_init.transpose(1, 0)), color=obj_colors[0])
    # 子目标点云
    for i in range(obj_subgoals.shape[0]):
        # current object pcd
        obj_pos = obj_subgoals[i, :3]
        obj_qua = obj_subgoals[i, 3:7]
        object_pcd_sg = tf.transPts_tq(object_pcd, obj_pos, obj_qua)
        ax.scatter(*tuple(object_pcd_sg.transpose(1, 0)), color=obj_colors[i+1])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_zlim(0.79, 1.05)
    plt.xticks(np.arange(-0.3, 0.3, 0.05))
    plt.yticks(np.arange(-0.3, 0.3, 0.05))
    plt.show()


def visual_relative_subgoal_nonprehensile(
        raw_obs: dict, 
        object_subgoals: np.ndarray, 
        finger_subgoals: np.ndarray,
        object_pcd: np.ndarray,
        scene_pcd,
        finger_pcd: np.ndarray=None,
        ):
    """
    visual current and subgoal pose/position of object and finger
    其中的物体子目标位姿是相对于物体的初始状态的新坐标系
    args:
        - raw_obs: h5py dict {
            - object_pos
            - object_quat
            - eef_pos
            - eef_quat
            - fingers_position
        }
        obs['object'] start with object pose, shape=(N, S) N为当前轨迹长度，S为state维度
        - obj_sub_goals: object pose subgoals, np.adarray, shape=(N, 7) pos+quat
        - finger_sub_goals: finger position subgoals, np.adarray, shape=(N, 6), second dim denotes left and right fingers
        - object_pcd: object pointcloud, shape=(n, 3)
        - finger_pcd: finger pointcloud, shape=(n, 3), if not set, use default sphere pointcloud
    """
    if finger_pcd is None:
        finger_radius = 0.008
        ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
        finger_pcd = np.asarray(ft_mesh.vertices)

    obj_pos_init = raw_obs['object_pos'][0]
    obj_qua_init = raw_obs['object_quat'][0]
    object_pcd_init = tf.transPts_tq(object_pcd, obj_pos_init, obj_qua_init)
    # 物体初始状态下的坐标系：原点位于点云质心，坐标轴与世界坐标系平行
    object_pcd_center = np.mean(object_pcd_init, axis=0) # (n, 3)
    T_W_Oc = tf.PosEuler_to_TransMat(object_pcd_center, (0, 0, 0))
    # 新坐标系下的物体点云
    object_pcd_Oc = object_pcd_init - object_pcd_center
    
    object_pcd_sequence = list()
    object_pcd_sequence.append(object_pcd_init) # (n, 1024, 3)
    for i in range(object_subgoals.shape[0]):
        # 计算各子目标物体位姿下的点云
        # P_W_Oss
        T_Oc_Oss = tf.PosQua_to_TransMat(object_subgoals[i, :3], object_subgoals[i, 3:])
        T_W_Oss = np.matmul(T_W_Oc, T_Oc_Oss)
        object_pcd_subgoal = tf.transPts_T(object_pcd_Oc, T_W_Oss)
        object_pcd_sequence.append(object_pcd_subgoal)
        
    object_pcd_sequence = np.concatenate(tuple(object_pcd_sequence), axis=-1)
    visual_object_pcd_sequence(scene_pcd, object_pcd_sequence)


def visual_object_pcd_sequence(
        scene_pcd, 
        object_pcd_sequence):
    """
    visual batch data in a point cloud

    args:
        - scene_pcd: scene point cloud (n, 3)
        - object_pcd_sequence: (n, c)
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')

    # scene
    ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

    # object
    object_pcds = object_pcd_sequence
    pcd_n = object_pcds.shape[1]//3
    assert pcd_n*3 == object_pcds.shape[1]
    obj_colors = Color.gradient_colors(pcd_n, start_c=[255, 0, 0], end_c=[0, 0, 255])
    for i in range(pcd_n):
        obj_pcd = object_pcds[:, (3*i):(3*(i+1))]
        ax.scatter(*tuple(obj_pcd.transpose(1, 0)), color=obj_colors[i])

    # show
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # plt.savefig('savefig_example.png')
    plt.gca().set_box_aspect((2, 2, 1))  # 当x、y、z轴范围之比为3:5:2时。
    plt.show()

def getFingersPos(eef_pos, eef_quat, lf_d, rf_d):
    """
    获取机械手两手指末端的3D坐标,相对于世界坐标系

    args:
        eef_pos (np.array): 机械臂末端位置, obs['robot0_eef_pos']
        eef_quat (np.array): 机械臂末端四元数, obs['robot0_eef_quat']
        gripper_width (float): 机械手张开宽度

    return:
        p_W_fl (np.array): 手指1在世界坐标系中的3D坐标
        p_W_fr (np.array): 手指2在世界坐标系中的3D坐标
    """
    # 机械手坐标系：在初始状态时，z轴向下，y轴向左，x轴向屏幕前
    # 计算T_B_E
    rot_mat = tf.quaternion_to_rotation_matrix(eef_quat)
    T_W_E = tf.PosRmat_to_TransMat(eef_pos, rot_mat)
    # p_E_f
    p_E_fl = np.array([0, lf_d, 0, 1]).reshape(-1, 1)
    p_E_fr = np.array([0, rf_d, 0, 1]).reshape(-1, 1)
    # p_B_f = T_B_E * p_E_f
    p_W_fl = np.matmul(T_W_E, p_E_fl)
    p_W_fr = np.matmul(T_W_E, p_E_fr)
    p_W_fl = p_W_fl.flatten()[:3]
    p_W_fr = p_W_fr.flatten()[:3]

    return p_W_fl, p_W_fr


def getGripperPos(eef_pos, eef_quat):
    """
    获取机械手的位置，为自定义的位置，位于eef的z轴正方向，距离为-0.01，用来指示eef的z轴正方向

    args:
        eef_pos (np.array): 机械臂末端位置, obs['robot0_eef_pos']
        eef_quat (np.array): 机械臂末端四元数, obs['robot0_eef_quat']

    return:
        p_W_fl (np.array): 械手的位置
    """
    # 机械手坐标系：在初始状态时，z轴向下，y轴向左，x轴向屏幕前
    # 计算T_B_E
    rot_mat = tf.quaternion_to_rotation_matrix(eef_quat)
    T_W_E = tf.PosRmat_to_TransMat(eef_pos, rot_mat)
    # p_E_f
    p_E_g = np.array([0, 0, -0.01, 1]).reshape(-1, 1)
    # p_B_f = T_B_E * p_E_f
    p_W_g = np.matmul(T_W_E, p_E_g)
    p_W_g = p_W_g.flatten()[:3]

    return p_W_g


class Color:
    scale = 255.
    base_color = {
        'r': [255, 0, 0],
        'red': [255, 0, 0],
        'g': [0, 255, 0],
        'green': [0, 255, 0],
        'b': [0, 0, 255],
        'blue': [0, 0, 255],
        # 'b': [0, 0, 0],
        'black': [0, 0, 0]
    }
    
    # def __init__(self) -> None:
        # pass 
    
    @classmethod
    def color(self, c:str) -> np.ndarray:
        """
        c: 颜色str
        return: rgb
        """
        assert c in self.base_color.keys()
        return np.array(self.base_color[c])

    @classmethod
    def gradient_colors(
            self, 
            num: int,
            start_c: Union[str, list, np.ndarray], 
            end_c: Union[str, list, np.ndarray], 
            ) -> np.ndarray:
        """
        返回渐变的颜色序列

        args:
            - num: 需要的颜色序列的长度
            - start_c: 序列第一个颜色，可以是str, list, np.ndarray
                - str: r, g, b, red, green, blue
                - list: rgb
                - np.ndarray: one-dim rgb
            - end_c: 序列最后一个颜色，格式和start_c一样
        
        return:
            - colors: rgb颜色序列, two-dim np.ndarray, shape=(num, 3)
        """
        assert num > 1

        if isinstance(start_c, str):
            assert start_c in self.base_color.keys()
            start_color = np.array(self.base_color[start_c])
        else:
            start_color = np.array(start_c)

        if isinstance(end_c, str):
            assert end_c in self.base_color.keys()
            end_color = np.array(self.base_color[end_c])
        else:
            end_color = np.array(end_c)

        step = (end_color - start_color) / (num-1)  # (3,)
        step_num = np.arange(num)[:, np.newaxis]   # (num, 1)
        steps = np.expand_dims(step, axis=0).repeat(num, axis=0)    # (num, 3)
        colors = steps * step_num + start_color
        colors[-1] = end_color
        return colors / self.scale
    
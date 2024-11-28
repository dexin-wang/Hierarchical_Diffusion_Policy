from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import shapely.ops
import math
import open3d as o3d
import torch
import scipy
import trimesh
from hiera_diffusion_policy.so3diffusion.util import quat_to_rmat


def subgoal_direct_detection_v1(pt1, pt2, pcd, radius=0.008):  #! 5mm->8mm
    """
    判断以pt1和pt2相连的线段为中心的圆柱是否与pcd有交集，radius为圆柱半径
    return: 
        True: 无交集
        False: 有交集
    """
    assert len(pcd.shape) == 2 and pcd.shape[1] == 3

    t1 = time.time()
    # 计算圆柱的旋转
    vec0 = np.array([0, 0, 1])
    vec1 = pt2 - pt1
    # 计算两向量的夹角
    tem = vec0.dot(vec1)
    tep = np.sqrt(vec0.dot(vec0) * vec1.dot(vec1))
    angle = np.arccos(tem / tep)
    if np.isnan(angle):
        angle = np.arccos(tep / tem)
    
    if angle == 0:
        rot_matrix = np.eye(3)
    else:
        # 向量叉乘得旋转轴
        axis = np.cross(vec0, vec1)
        # 轴角转旋转矩阵
        rot_matrix = scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * angle))
    transform = PosRmat_to_TransMat((pt1+pt2)/2, rot_matrix)
    # 构建mesh
    h = np.linalg.norm(pt1-pt2)
    mesh = trimesh.primitives.Cylinder(
        radius=radius, 
        height=h,
        transform=transform)
    #* 可视化圆柱
    # mesh.show()
    t2 = time.time()
    
    res = max(trimesh.proximity.signed_distance(mesh, pcd)) < 0

    # print('构建mesh耗时', t2-t1)
    # print('计算碰撞耗时', time.time()-t2)
    return res


def subgoal_direct_detection(pt1, pt2, pcd, radius):
    """
    判断以pt1和pt2相连的线段为中心的圆柱是否与pcd有交集，radius为圆柱半径

    在pt1和pt2的连线上每隔1mm取一个点，判断改点与点云的距离是否小于radius
    pcd: (N, 3)
    return: 
        True: 无交集
        False: 有交集
    """
    assert len(pcd.shape) == 2 and pcd.shape[1] == 3

    length = np.linalg.norm(pt2-pt1)
    p_num = int(length/0.001)
    if p_num <= 0:
        return True
    
    for i in range(p_num):
        # 计算点的位置
        p = pt1 + (pt2-pt1)/length*0.001*i
        # 检测点与点云的距离
        dists = np.sqrt(np.sum(np.square(pcd-p), axis=1))    # (N,)
        if np.min(dists) < radius: 
            return False
    return True


def check_fin_obj_collision(obj_pcd, f_pos, radius):
    """
    检测手指模型与物体点云是否有碰撞

    obj_pcd: 物体点云 (n, 3)
    f_pos: 手指位置 (3,)
    radius: 手指半径 float

    return: 
        True: 有碰撞
        False: 无碰撞
    """
    # 构建手指mesh
    mesh = trimesh.primitives.Sphere(radius, f_pos)
    #* 可视化圆柱
    # mesh.show()
    res = max(trimesh.proximity.signed_distance(mesh, obj_pcd)) > 0
    return res

def angle_3dvector(vec0, vec1):
    # 计算两向量的夹角(弧度)
    tem = vec0.dot(vec1)
    tep = np.sqrt(vec0.dot(vec0) * vec1.dot(vec1))
    angle = np.arccos(tem / tep)
    if np.isnan(angle):
        angle = np.arccos(tep / tem)
    return angle


def qua_diff(qua1, qua2):
    """
    计算两个四元数的夹角，弧度
    """
    dot_product = np.sum(np.array(qua1) * np.array(qua2))
    dot_product = np.clip(dot_product, -1, 1)
    error = np.arccos(abs(dot_product))
    return error


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
    

def quaternion_to_rotation_matrix(q):
    """
    四元数转旋转矩阵
    xyzw
    """
    return R.from_quat(q).as_matrix()

def quaternion_to_rotation_matrix_npBatch(qs):
    """
    四元数转旋转矩阵
    xyzw
    qs: (n, 4)
    """
    rs = list()
    for q in qs:
        r = R.from_quat(q).as_matrix()
        rs.append(r) 
    return np.array(rs)

def PosRmat_to_TransMat(pos, rot):
    """
    将位置和旋转矩阵转换为转换矩阵

    input:
        pos: [ndarray, (3,), np.float]
            xyz坐标
        rot: [ndarray, (3,3), np.float]
            旋转矩阵

    return:
        T: [ndarray, (4,4), np.float]
            转换矩阵
    """
    T = np.zeros((4, 4), dtype=float)
    T[:3, :3] = rot
    T[:, 3] = np.r_[np.array(pos), np.array([1,])]
    return T

def PosRmat_to_TransMat_batch(pos: torch.Tensor, rmat: torch.Tensor):
    """
    将位置和旋转矩阵转换为转换矩阵

    input:
        - pos: [B, ..., 3]
        - rmat: [B, ..., 3, 3] 旋转矩阵

    return:
        T: (B, ..., 4, 4)
            转换矩阵
    """
    B = pos.shape[0]
    T = torch.zeros(rmat.shape[:-2] + (4, 4), device=pos.device)
    T[..., :3, :3] = rmat
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1
    return T

def PosRmat_to_TransMat_npbatch(pos: np.ndarray, rmat: np.ndarray):
    """
    将位置和旋转矩阵转换为转换矩阵

    input:
        - pos: [B, ..., 3]
        - rmat: [B, ..., 3, 3] 旋转矩阵

    return:
        T: (B, ..., 4, 4)
            转换矩阵
    """
    B = pos.shape[0]
    T = np.zeros(rmat.shape[:-2] + (4, 4))
    T[..., :3, :3] = rmat
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1
    return T

def PosRmat_to_TransMat_batch_v1(pos: torch.Tensor, rmat: torch.Tensor):
    """
    将位置和旋转矩阵转换为转换矩阵

    input:
        - pos: [B, ..., 3]
        - rmat: [B, ..., 3, 3] 旋转矩阵

    return:
        T: (B, ..., 4, 4)
            转换矩阵
    """
    B = pos.shape[0]
    T = torch.zeros((B, 4, 4), device=pos.device)
    T[:, :3, :3] = rmat
    T[:, :3, 3] = pos
    T[:, 3, 3] = 1
    return T


def PosQua_to_TransMat(pos, qua):
    """
    将位置和四元数转换为旋转矩阵
    input:
        - pos: [ndarray, (3,), np.float]
            xyz坐标
        - qua: [ndarray, (4,), np.float]
            xyzw四元数
    """
    return PosRmat_to_TransMat(pos, quaternion_to_rotation_matrix(qua))


def PosQua_to_TransMat_batch(pos: torch.Tensor, qua: torch.Tensor):
    """
    将位置和四元数转换为旋转矩阵
    input:
        - pos: (B, 3) xyz坐标
        - qua: (B, 4) xyzw四元数
    """
    qua_wxyz = torch.concat((qua[:, 3][...,None], qua[:, :3]), dim=1)   # wxyz
    return PosRmat_to_TransMat_batch(pos, quat_to_rmat(qua_wxyz)) # (B, 4, 4)


def PosQua_to_TransMat_npbatch(pos: np.ndarray, qua: np.ndarray):
    """
    将位置和四元数转换为旋转矩阵

    input:
        - pos: (B, 3) xyz坐标
        - qua: (B, 4) xyzw四元数
    
    return:
        - Tmat: (B, 4, 4)
    """
    rmats = list()
    for q in qua:
        rmats.append(quaternion_to_rotation_matrix(q))
    rmats = np.array(rmats)
    return PosRmat_to_TransMat_npbatch(pos, rmats) # (B, 4, 4)

def Qua_to_Euler(qua):
    """
    四元数转欧拉角
    qua: (4,)
    return: euler (3,)
    """
    return R.from_quat(qua).as_euler('xyz', degrees=False)

def Euler_to_Qua(euler):
    """
    欧拉角转四元数
    eulers: (3,)
    return: qua (4,)
    """
    return R.from_euler(seq='xyz', angles=euler, degrees=False).as_quat()

def Euler_to_Qua_npbatch(eulers):
    """
    欧拉角转四元数
    eulers: (B, 3)
    return: qua (B, 4)
    """
    qs = list()
    for euler in eulers:
        qs.append(R.from_euler(seq='xyz', angles=euler, degrees=False).as_quat())
    return np.array(qs)


def PosEuler_to_TransMat(pos, euler):
    """
    将位置和欧拉角转换为转换矩阵

    input:
        pos: [ndarray, (3,), np.float]
            xyz坐标
        euler: [ndarray, (3,), np.float]
            欧拉角

    return:
        T: [ndarray, (4,4), np.float]
            转换矩阵
    """
    return PosRmat_to_TransMat(
        pos, R.from_euler(seq='xyz', angles=euler, degrees=False).as_matrix())


def TransMat_to_PosQua(TMat):
    """
    将转换矩阵转为位置和四元数

    input:
        TMat: 转换矩阵 (4, 4)
    
    return:
        pos: xyz坐标 (3,)
        qua: xyzw四元数 (4,)
    """
    rmat = TMat[:3, :3]
    pos = TMat[:3, 3]
    qua = R.from_matrix(rmat).as_quat()
    return pos, qua


def TransMat_to_PosQua_npbatch(TMat):
    """
    将转换矩阵转为位置和四元数

    input:
        TMat: 转换矩阵 (B, 4, 4)
    
    return:
        pos: xyz坐标 (B, 3,)
        qua: xyzw四元数 (B, 4)
    """
    rmat = TMat[:, :3, :3]  # (B, 3, 3)
    pos = TMat[:, :3, 3]    # (B, 3)
    qua = list()
    for r in rmat:
        qua.append(R.from_matrix(r).as_quat())
    return pos, np.array(qua)


def transPts_tq(P_f1_pts, t_f2_f1, q_f2_f1):
    """
    将f1坐标系下的点集pts，转换到f2坐标系下

    args:
        - P_f1_pts: (n, 3) 点集pts在f1坐标系下的坐标
        - t_f2_f1: f2坐标系到f1坐标系的平移
        - q_f2_f1: f2坐标系到f1坐标系的四元数

    return:
        - pts: (n, 3)
    """
    T_f2_f1 = PosRmat_to_TransMat(t_f2_f1, quaternion_to_rotation_matrix(q_f2_f1))
    one = np.ones((1, P_f1_pts.shape[0]))
    _P_f1_pts = np.concatenate((P_f1_pts.T, one), axis=0)   # (4,4)
    P_f2_f1 = np.matmul(T_f2_f1, _P_f1_pts)
    return P_f2_f1.T[:, :3]

def transPts_tq_npbatch(BP_f1_pts:np.ndarray, Bt_f2_f1:np.ndarray, Bq_f2_f1:np.ndarray=None):
    """
    将f1坐标系下的点集pts，转换到f2坐标系下

    args:
        - BP_f1_pts: (B, n, 3) 点集pts在f1坐标系下的坐标
        - t_f2_f1: (B, 3) f2坐标系到f1坐标系的平移
        - q_f2_f1: (B, 4) f2坐标系到f1坐标系的四元数

    return:
        - pts: (N, n, 3)
    """
    Bpts = list()
    for i in range(BP_f1_pts.shape[0]):
        if Bq_f2_f1 is None:
            q = (0, 0, 0, 1)
        else:
            q = Bq_f2_f1[i]
        pts = transPts_tq(BP_f1_pts[i], Bt_f2_f1[i], q)
        Bpts.append(pts)
    return np.array(Bpts)

def transPts_T(P_f1_pts, T_f2_f1):
    """
    将f1坐标系下的点集pts，转换到f2坐标系下

    args:
        - P_f1_pts: (n, 3) 点集pts在f1坐标系下的坐标
        - t_f2_f1: f2坐标系到f1坐标系的平移
        - q_f2_f1: f2坐标系到f1坐标系的四元数

    return:
        - pts: (n, 3)
    """
    one = np.ones((1, P_f1_pts.shape[0]))
    _P_f1_pts = np.concatenate((P_f1_pts.T, one), axis=0)   # (4,4)
    P_f2_f1 = np.matmul(T_f2_f1, _P_f1_pts)
    return P_f2_f1.T[:, :3]


def transPts_T_npbatch(P_f1_pts: np.ndarray, T_f2_f1: np.ndarray):
    """
    将f1坐标系下的点集pts，转换到f2坐标系下

    args:
        - P_f1_pts: (B n, 3) 点集pts在f1坐标系下的坐标
        - T_f2_f1: (B, 4, 4) f2坐标系到f1坐标系的转换矩阵

    return:
        - pts: (B, ..., n, 3)
    """
    one = np.ones(T_f2_f1.shape[:-2] + (1, P_f1_pts.shape[-2]))  # (B, 1, n)
    _P_f1_pts = np.concatenate((P_f1_pts.transpose(0, 2, 1), one), axis=1)  # (B, 4, n)
    P_f2_f1 = np.matmul(T_f2_f1, _P_f1_pts)
    return P_f2_f1.transpose(0, 2, 1)[..., :3]


def transPts_T_batch(P_f1_pts: torch.Tensor, T_f2_f1: torch.Tensor):
    """
    将f1坐标系下的点集pts，转换到f2坐标系下

    args:
        - P_f1_pts: (B, ..., n, 3) 点集pts在f1坐标系下的坐标
        - T_f2_f1: (B, ..., 4, 4) f2坐标系到f1坐标系的转换矩阵

    return:
        - pts: (B, ..., n, 3)
    """
    one = torch.ones(T_f2_f1.shape[:-2] + (1, P_f1_pts.shape[-2]), device=P_f1_pts.device)  # (B, ..., 1, n)
    _P_f1_pts = torch.concat((P_f1_pts.transpose(-1, -2), one), dim=-2)  # (B, ..., 4, n)
    P_f2_f1 = torch.matmul(T_f2_f1, _P_f1_pts)
    return P_f2_f1.transpose(-1, -2)[..., :3]


def transPts_T_batch_v1(P_f1_pts: torch.Tensor, T_f2_f1: torch.Tensor):
    """
    将f1坐标系下的点集pts，转换到f2坐标系下

    args:
        - P_f1_pts: (B, n, 3) 点集pts在f1坐标系下的坐标
        - T_f2_f1: (B, 4, 4) f2坐标系到f1坐标系的转换矩阵

    return:
        - pts: (B, n, 3)
    """
    B = P_f1_pts.shape[0]
    one = torch.ones((B, 1, P_f1_pts.shape[1]), device=P_f1_pts.device)
    _P_f1_pts = torch.concat((P_f1_pts.transpose(-1, -2), one), dim=1)  # (B, 4, n)
    P_f2_f1 = torch.matmul(T_f2_f1, _P_f1_pts)
    return P_f2_f1.transpose(-1, -2)[..., :3]


def transPt(P_f1_pt, T_f2_f1=None, t_f2_f1=None, q_f2_f1=None):
    """
    将f1坐标系下的点pt，转换到f2坐标系下

    args:
        - P_f1_pt: (3,) 点pt在f1坐标系下的坐标
        - t_f2_f1: f2坐标系到f1坐标系的平移
        - q_f2_f1: f2坐标系到f1坐标系的四元数

    return:
        - pts: (3,)
    """
    if T_f2_f1 is not None:
        pass
    elif t_f2_f1 is not None and q_f2_f1 is not None:
        T_f2_f1 = PosRmat_to_TransMat(t_f2_f1, quaternion_to_rotation_matrix(q_f2_f1))
    else:
        raise ValueError

    _P_f1_pt = np.array([
        [P_f1_pt[0]], 
        [P_f1_pt[1]], 
        [P_f1_pt[2]],
        [1]
    ])
    P_f2_pt = np.matmul(T_f2_f1, _P_f1_pt) # (4, 1)
    return P_f2_pt[:3, 0]


def transPt_tq_batch(P_f1_pt, t_f2_f1=None, q_f2_f1=None):
    """
    将f1坐标系下的点pt，转换到f2坐标系下

    args:
        - P_f1_pt: (B, 3) 点pt在f1坐标系下的坐标
        - t_f2_f1: (B, 3) f2坐标系到f1坐标系的平移
        - q_f2_f1: (B, 4) f2坐标系到f1坐标系的四元数 xyzw

    return:
        - pts: (B, 3)
    """
    # quaternion to rotation
    qua = torch.concat((q_f2_f1[:, 3][...,None], q_f2_f1[:, :3]), dim=1)   # wxyz
    T_f2_f1 = PosRmat_to_TransMat_batch(t_f2_f1, quat_to_rmat(qua)) # (B, 4, 4)

    _P_f1_pt = torch.ones((P_f1_pt.shape[0], 4, 1), device=P_f1_pt.device)  # (B, 4, 1)
    _P_f1_pt[:, :3] = P_f1_pt   # (B, 3, 1)
    P_f2_pt = torch.matmul(T_f2_f1, _P_f1_pt)   # (B, 4, 1)
    return P_f2_pt[:, :3, 0]

def transPt_T_batch(P_f1_pt, T_f2_f1):
    """
    将f1坐标系下的点pt，转换到f2坐标系下

    args:
        - P_f1_pt: (B, 3) 点pt在f1坐标系下的坐标
        - T_f2_f1: (B, 4, 4) f2坐标系到f1坐标系的转换矩阵

    return:
        - pts: (B, 3)
    """
    _P_f1_pt = torch.ones((P_f1_pt.shape[0], 4, 1), device=P_f1_pt.device)  # (B, 4, 1)
    _P_f1_pt[:, :3, 0] = P_f1_pt   # (B, 3, 1)
    P_f2_pt = torch.matmul(T_f2_f1, _P_f1_pt)   # (B, 4, 1)
    return P_f2_pt[:, :3, 0]


def transPt_T_npbatch(
        P_f1_pt: np.ndarray, 
        T_f2_f1: np.ndarray):
    """
    将f1坐标系下的点pt，转换到f2坐标系下

    args:
        - P_f1_pt: (B, ..., 3) 点pt在f1坐标系下的坐标
        - T_f2_f1: (B, ..., 4, 4) f2坐标系到f1坐标系的转换矩阵

    return:
        - pts: (B, 3)
    """
    _P_f1_pt = np.ones(P_f1_pt.shape[:-1]+(4, 1))  # (B, ..., 4, 1)
    _P_f1_pt[..., :3, 0] = P_f1_pt   # (B, ..., 3, 1)
    P_f2_pt = np.matmul(T_f2_f1, _P_f1_pt)   # (B, ..., 4, 1)
    return P_f2_pt[..., :3, 0]

# def TransMat_to_PosRmat(transMat):
#     """
#     将转换矩阵解码为位置+四元数
#     transMat: 转换矩阵, shape=(4,4)
#     """
#     pos = transMat[:3, 3]
#     rmat = transMat[:3, :3]
#     return transMat[:3, 3], transMat[:3, :3]


def create_point_cloud(im_rgb, im_dep, cameraInMatrix, workspace_mask=None):
    """ Generate point cloud using depth image only.

        Input:
            im_rgb: [numpy.ndarray, (H,W,3), numpy.float32]
                rgb image
            im_dep: [numpy.ndarray, (H,W), numpy.float32]
                depth image 单位m
            cameraInMatrix: 相机内参

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    # 生成点云
    xmap = np.arange(im_rgb.shape[1])
    ymap = np.arange(im_rgb.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = im_dep
    points_x = (xmap - cameraInMatrix[0][2]) * points_z / cameraInMatrix[0][0]
    points_y = (ymap - cameraInMatrix[1][2]) * points_z / cameraInMatrix[1][1]
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if workspace_mask is not None:
        cloud = cloud[workspace_mask]
    cloud = cloud.reshape([-1, 3])

    return cloud

def removeOutLier_pcl(pcl, nb_points=10, radius=0.02):
    """
    去除点云中的离群点
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    res = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)#半径方法剔除
    pcd = res[0]#返回点云，和点云索引
    return np.asarray(pcd.points)

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


# ================= 二维位姿变换 =================
def PosRad_to_Tmat(pos, rad):
    """将二维坐标和弧度旋转角转换为齐次矩阵
    按照x->y的旋转格式
    """
    R = np.array([[np.cos(rad), -np.sin(rad), pos[0]],
                  [np.sin(rad),  np.cos(rad), pos[1]],
                  [0,0,1]])
    return R

def transPt2D(P_f1_pt, T_f2_f1):
    """二维空间中的坐标点变换
    P_f1_pt: (2,)
    T_f2_f1: (3, 3)
    """
    _P_f1_pt = np.array([
        [P_f1_pt[0]], 
        [P_f1_pt[1]], 
        [1]
    ])
    P_f2_pt = np.matmul(T_f2_f1, _P_f1_pt) # (3, 1)
    return P_f2_pt[:2, 0]
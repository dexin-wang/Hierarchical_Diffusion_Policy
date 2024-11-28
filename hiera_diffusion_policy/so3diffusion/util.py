import inspect
from collections import namedtuple
from typing import Tuple, Iterable
from math import sqrt, log, exp
from itertools import product

import torch


class AffineT(object):
    def __init__(self, rot: torch.Tensor, shift: torch.Tensor):
        super().__init__()
        self.rot = rot
        self.shift = shift

    def __len__(self):
        return max(len(self.rot), len(self.shift))

    def __getitem__(self, item):
        return AffineT(self.rot[item], self.shift[item])

    @property
    def device(self):
        return self.rot.device

    @property
    def shape(self):
        return self.shift.shape

    def to(self, device):
        self.rot.to(device)
        self.shift.to(device)
        return AffineT(self.rot.to(device), self.shift.to(device))

    @classmethod
    def from_euler(cls, euls: torch.Tensor, shift: torch.Tensor):
        rot = euler_to_rmat(*torch.unbind(euls, dim=-1))
        return cls(rot, shift)

    def detach(self):
        d_rot = self.rot.detach()
        d_shift = self.shift.detach()
        return AffineT(d_rot, d_shift)


class AffineGrad(object):
    def __init__(self, rot_g, shift_g):
        super().__init__()
        self.rot_g = rot_g
        self.shift_g = shift_g

    def __len__(self):
        return max(len(self.rot_g), len(self.shift_g))

    def __getitem__(self, item):
        return AffineGrad(self.rot_g[item], self.shift_g[item])


ProtData = namedtuple("ProtData", ['residues', 'positions', 'angles'])


def rmat2six(x: torch.Tensor) -> torch.Tensor:
    # Drop last column
    return torch.flatten(x[..., :2, :], -2, -1)


def six2rmat(x: torch.Tensor) -> torch.Tensor:
    a1 = x[..., :3]
    a2 = x[..., 3:6]
    b1 = a1 / a1.norm(p=2, dim=-1, keepdim=True)
    b1_a2 = (b1 * a2).sum(dim=-1, keepdim=True)  # Dot product
    b2 = (a2 - b1_a2 * b1)
    b2 = b2 / b2.norm(p=2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    out = torch.stack((b1, b2, b3), dim=-2)
    return out


def skew2vec(skew: torch.Tensor) -> torch.Tensor:
    """
    skew: 斜对称矩阵, (B, 3, 3)
    return: v, (B, 3)
    """
    vec = torch.zeros_like(skew[..., 0])
    vec[..., 0] = skew[..., 2, 1]
    vec[..., 1] = -skew[..., 2, 0]
    vec[..., 2] = skew[..., 1, 0]
    return vec


def vec2skew(vec: torch.Tensor) -> torch.Tensor:
    skew = torch.repeat_interleave(torch.zeros_like(vec).unsqueeze(-1), 3, dim=-1)  # (B, 3, 3) 全0
    skew[..., 2, 1] = vec[..., 0]
    skew[..., 2, 0] = -vec[..., 1]
    skew[..., 1, 0] = vec[..., 2]
    return skew - skew.transpose(-1, -2)


def orthogonalise(mat):
    """Orthogonalise rotation/affine matrices

    Ideally, 3D rotation matrices should be orthogonal,
    however during creation, floating point errors can build up.
    We SVD decompose our matrix as in the ideal case S is a diagonal matrix of 1s
    We then round the values of S to [-1, 0, +1],
    making U @ S_rounded @ V.T an orthonormal matrix close to the original.
    """
    orth_mat = mat.clone()
    u, s, v = torch.svd(mat[..., :3, :3])
    orth_mat[..., :3, :3] = u @ torch.diag_embed(s.round()) @ v.transpose(-1, -2)
    return orth_mat


def rmat_cosine_dist(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    ''' Calculate the cosine distance between two (batched) rotation matrices

    '''
    # rotation matrices have A^-1 = A_transpose
    # multiplying the inverse of m2 by m1 gives us a combined transform
    # corresponding to roting by m1 and then back by the *opposite* of m2.
    # if m1 == m2, m_comb = identity
    m_comb = m2.transpose(-1, -2) @ m1
    # Batch trace calculation
    tra = torch.einsum('...ii', m_comb)
    # The trace of a rotation matrix = 1+2*cos(a)
    # where a is the angle in the axis-angle representation
    # Cosine dist is defined as 1 - cos(a)
    out = 1 - (tra - 1) / 2
    return out


def rmat_gaussian_kernel(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    ''' Calculate the gaussian kernel between two (batched) rotation matrices

    '''
    dist = rmat_dist(m1,m2)

    return torch.exp(-dist)

def rmat_cosine_kernel(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    ''' Calculate the cosine kernel between two (batched) rotation matrices

    '''
    # rotation matrices have A^-1 = A_transpose
    # multiplying the inverse of m2 by m1 gives us a combined transform
    # corresponding to roting by m1 and then back by the *opposite* of m2.
    # if m1 == m2, m_comb = identity
    m_comb = m2.transpose(-1, -2) @ m1
    # Batch trace calculation
    tra = torch.einsum('...ii', m_comb)
    # The trace of a rotation matrix = 1+2*cos(a)
    # where a is the angle in the axis-angle representation
    # Cosine dist is defined as 1 - cos(a)
    out = (tra - 1) / 2
    return out


# See paper
# Exponentials of skew-symmetric matrices and logarithms of orthogonal matrices
# https://doi.org/10.1016/j.cam.2009.11.032
# For most of the derivatons here

# We use atan2 instead of acos here dut to better numerical stability.
# it means we get nicer behaviour around 0 degrees
# More effort to derive sin terms
# but as we're dealing with small angles a lot,
# the tradeoff is worth it.
def log_rmat(r_mat: torch.Tensor) -> torch.Tensor:
    """
    rmat: (B, 3, 3) Batch个沿z轴旋转90或-90度的旋转矩阵
    #! 符号和论文公式相反，但结果正确
    """
    #* 计算log(R)
    skew_mat = (r_mat - r_mat.transpose(-1, -2))    #! 和论文描述的符号相反
    sk_vec = skew2vec(skew_mat) #! 和论文描述的符号相反，负负得正
    s_angle = (sk_vec).norm(p=2, dim=-1) / 2    # sin(theta)
    c_angle = (torch.einsum('...ii', r_mat) - 1) / 2    # '...ii' 计算矩阵的迹， cos(theta)
    angle = torch.atan2(s_angle, c_angle)   # (B,)
    scale = (angle / (2 * s_angle)) # log(R)的系数 (B,)
    # if s_angle = 0, i.e. rotation by 0 or pi (180), we get NaNs
    # by definition, scale values are 0 if rotating by 0.
    # This also breaks down if rotating by pi, fix further down
    scale[angle == 0.0] = 0.0
    log_r_mat = scale[..., None, None] * skew_mat   #! 和论文描述的符号相反

    # Check for NaNs caused by 180deg rotations.
    nanlocs = log_r_mat[...,0,0].isnan()    # (B,)
    nanmats = r_mat[nanlocs]    # (n, 3, 3)
    # 对于180度的旋转，使用其他方式计算log
    # We need to use an alternative way of finding the logarithm for nanmats,
    # Use eigendecomposition to discover axis of rotation.
    # By definition, these are symmetric, so use eigh.
    # NOTE: linalg.eig() isn't in torch 1.8,
    #       and torch.eig() doesn't do batched matrices
    eigval, eigvec = torch.linalg.eigh(nanmats) # 矩阵特征分解 eigvec是正交矩阵
    # Final eigenvalue == 1, might be slightly off because floats, but other two are -ve.
    # this *should* just be the last column if the docs for eigh are true.
    nan_axes = eigvec[...,-1,:] # 最后一行 单位向量
    nan_angle = angle[nanlocs]  # 角度 (n,)
    nan_skew = vec2skew(nan_angle[...,None] * nan_axes)
    log_r_mat[nanlocs] = nan_skew
    return log_r_mat    #! 和论文描述的符号相反


def aa_to_rmat(rot_axis: torch.Tensor, ang: torch.Tensor):
    '''Generates a rotation matrix (3x3) from axis-angle form

        `rot_axis`: Axis to rotate around, defined as vector from origin.
        `ang`: rotation angle
        '''
    rot_axis_n = rot_axis / rot_axis.norm(p=2, dim=-1, keepdim=True)
    sk_mats = vec2skew(rot_axis_n)
    log_rmats = sk_mats * ang[..., None]
    rot_mat = torch.matrix_exp(log_rmats)
    return orthogonalise(rot_mat)


def rmat_to_aa(r_mat) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Calculates axis and angle of rotation from a rotation matrix.

        returns angles in [0,pi] range.

        `r_mat`: rotation matrix.
        '''
    log_mat = log_rmat(r_mat)
    skew_vec = skew2vec(log_mat)
    angle = skew_vec.norm(p=2, dim=-1, keepdim=True)
    axis = skew_vec / angle
    return axis, angle


def quat_to_rmat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Calculates rotation matrices from given quaternions

    returns rotation matrices shape (..., 3, 3).

    `quaternions`: quaternions with real part first, shape (..., 4).

    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            # row 0
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            # row 1
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            # row 2
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    # reshape to batches x 3 x 3
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def MMD(X: torch.Tensor, Y: torch.Tensor, kernel, chunksize=None):
    '''
    Calculate maximum mean descrepancy between two sets of tensors using a given kernel
    '''
    l_X = len(X)
    l_Y = len(Y)
    maxlen = max(l_X, l_Y)
    # As this involves an outer product, we can end up with matrices too big to fit in ram
    if chunksize is None or chunksize >= maxlen:
        X_sum = kernel(X.unsqueeze(0), X.unsqueeze(1)).sum(dim=(0, 1))
        Y_sum = kernel(Y.unsqueeze(0), Y.unsqueeze(1)).sum(dim=(0, 1))
        XY_sum = kernel(X.unsqueeze(0), Y.unsqueeze(1)).sum(dim=(0, 1))
    else:
        # If chunksize is set, split X and Y arrays into manageable lengths,
        splits = list(range(chunksize, maxlen, chunksize))
        X_split = torch.tensor_split(X, splits)
        Y_split = torch.tensor_split(Y, splits)
        # Take outer product kernel function thing and sum each chunk, then sum over them.
        X_chunk_sums = [kernel(x1.unsqueeze(0), x2.unsqueeze(1)).sum(dim=(0, 1)) for x1, x2 in product(X_split, X_split)]
        X_sum = sum(X_chunk_sums)

        Y_chunk_sums = [kernel(y1.unsqueeze(0), y2.unsqueeze(1)).sum(dim=(0, 1)) for y1, y2 in product(Y_split, Y_split)]
        Y_sum = sum(Y_chunk_sums)

        XY_chunk_sums = [kernel(x.unsqueeze(0), y.unsqueeze(1)).sum(dim=(0, 1)) for x, y in product(X_split, Y_split)]
        XY_sum = sum(XY_chunk_sums)

    X_ker_mean = (1 / (l_X ** 2)) * X_sum
    Y_ker_mean = (1 / (l_Y ** 2)) * Y_sum
    outer_mean = (2 / (l_X * l_Y)) * XY_sum

    return X_ker_mean + Y_ker_mean - outer_mean



def Ker_2samp_test(X, Y, kernel, alpha=0.05, max_ker=1, chunksize=None):
    '''Kernel two-sample test

    '''
    m = len(X)
    n = len(Y)
    assert m == n, "Requires equal amount of samples from X and Y"

    mmd = MMD(X,Y,kernel, chunksize=chunksize).item()
    test_val = (2*max_ker/m) ** 0.5 * (1 + (2*log(1/alpha))**0.5)
    return mmd < test_val

def Ker_2samp_log_prob(X, Y, kernel, max_ker=1, chunksize=None):
    '''Kernel two-sample test

        returns the log_probability of a type I error
        i.e. The log of the p value
    '''
    m = len(X)
    n = len(Y)
    assert m == n, "Requires equal amount of samples from X and Y"

    mmd = MMD(X,Y,kernel, chunksize=chunksize).item()
    return -((((mmd/((2*max_ker/m) ** 0.5))-1)**2)/2)


def rmat_dist(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''Calculates the geodesic distance between two (batched) rotation matrices.

    '''
    mul = input.transpose(-1, -2) @ target
    log_mul = log_rmat(mul)
    out = log_mul.norm(p=2, dim=(-1, -2))
    return out  # Frobenius norm


def so3_lerp(rot_a: torch.Tensor, rot_b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    ''' Weighted interpolation between rot_a and rot_b

    '''
    # Treat rot_b = rot_a @ rot_c
    # rot_a^-1 @ rot_a = I
    # rot_a^-1 @ rot_b = rot_a^-1 @ rot_a @ rot_c = I @ rot_c
    # once we have rot_c, use axis-angle forms to lerp angle
    rot_c = rot_a.transpose(-1, -2) @ rot_b
    axis, angle = rmat_to_aa(rot_c)
    # once we have axis-angle forms, determine intermediate angles.
    i_angle = weight * angle
    rot_c_i = aa_to_rmat(axis, i_angle)
    return rot_a @ rot_c_i

def so3_bezier(*rots, weight):
    if len(rots) == 2:
        return so3_lerp(*rots, weight=weight)
    else:
        a = so3_bezier(rots[:-1], weight=weight)
        b = so3_bezier(rots[1:], weight=weight)
        return so3_lerp(a,b, weight=weight)


def so3_scale(rmat, scalars):
    '''Scale the magnitude of a rotation matrix,
    e.g. a 45 degree rotation scaled by a factor of 2 gives a 90 degree rotation.

    This is the same as taking matrix powers, but pytorch only supports integer exponents

    So instead, we take advantage of the properties of rotation matrices
    to calculate logarithms easily. and multiply instead.

    args:
        - rmat: (B, 3, 3) Batch个沿z轴旋转90或-90度的旋转矩阵
        - scalars: (B,) 与timestep对应的 sqrt_alphas_cumprod
    return:
        out: 缩放后的旋转矩阵
    '''
    logs = log_rmat(rmat)
    scaled_logs = logs * scalars[..., None, None]
    out = torch.matrix_exp(scaled_logs)
    return out


def se3_lerp(transf_a: AffineT, transf_b: AffineT, weight: torch.Tensor) -> AffineT:
    ''' Weighted interpolation between transf_a and transf_a

    '''
    # Treat rot_b = rot_a @ rot_c
    # rot_a^-1 @ rot_a = I
    # rot_a^-1 @ rot_b = rot_a^-1 @ rot_a @ rot_c = I @ rot_c
    # once we have rot_c, use axis-angle forms to lerp angle
    rot_a = transf_a.rot
    rot_b = transf_b.rot
    rot_lerps = so3_lerp(rot_a, rot_b, weight)
    shift_a = transf_a.shift
    shift_b = transf_b.shift
    shift_lerps = torch.lerp(shift_a, shift_b, weight)

    return AffineT(rot_lerps, shift_lerps)


def se3_scale(transf: AffineT, scalars) -> AffineT:
    rot_scaled = so3_scale(transf.rot, scalars)
    shift_scaled = transf.shift * scalars[..., None]
    return AffineT(rot_scaled, shift_scaled)


def rmat_to_euler(rmat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sy = torch.sqrt(rmat[..., 0, 0] * rmat[..., 0, 0] + rmat[..., 1, 0] * rmat[..., 1, 0])
    x = torch.atan2(rmat[..., 2, 1], rmat[..., 2, 2])
    y = torch.atan2(rmat[..., 2, 0], sy)
    z = torch.atan2(rmat[..., 1, 0], rmat[..., 0, 0])
    return x, y, z


def euler_to_rmat(x, y, z):
    R_x = torch.eye(3).repeat(*x.shape, 1, 1).to(x)
    cos_x = torch.cos(x)
    sin_x = torch.sin(x)
    R_x[..., 1, 1] = cos_x
    R_x[..., 1, 2] = -sin_x
    R_x[..., 2, 1] = sin_x
    R_x[..., 2, 2] = cos_x

    R_y = torch.eye(3).repeat(*y.shape, 1, 1).to(y)
    cos_y = torch.cos(y)
    sin_y = torch.sin(y)
    R_y[..., 0, 0] = cos_y
    R_y[..., 2, 0] = sin_y
    R_y[..., 0, 2] = -sin_y
    R_y[..., 2, 2] = cos_y

    R_z = torch.eye(3).repeat(*z.shape, 1, 1).to(z)
    cos_z = torch.cos(z)
    sin_z = torch.sin(z)
    R_z[..., 0, 0] = cos_z
    R_z[..., 0, 1] = -sin_z
    R_z[..., 1, 0] = sin_z
    R_z[..., 1, 1] = cos_z

    R = R_z @ R_y @ R_x

    return R


def to_device(device, *objects, non_blocking=False):
    gpu_objects = []
    for object in objects:
        if isinstance(object, torch.Tensor):
            gpu_objects.append(object.to(device, non_blocking=non_blocking))
        elif isinstance(object, ProtData):
            gpu_objects.append(ProtData(*to_device(device, *object)))
        elif isinstance(object, Iterable):
            gpu_objects.append(to_device(device, *object))
        else:
            raise RuntimeError(f"Cannot move object of type {type(object)} to {device}")
    return gpu_objects


def init_from_dict(argdict, *classes):
    """Passes dictionary arguments into multiple classes
    Missing and extra args/kwargs are ignored.

    This allows for a single dict (i.e. argparse) to define
    the state of multiple different objects
    (e.g. optimiser, network and dataloader).

    If classes share a name of one of the arguments and
    it is present in the dictionary as a key,
    then the same value will be used for both class init calls
    """

    objs = []
    for cls in classes:
        sig = inspect.signature(cls)
        args = [k for k, v in sig.parameters.items() if
                v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        class_kwargs = {k: v for k, v in argdict.items() if k in args}
        objs.append(cls(**class_kwargs))
    return objs


def identity(x):
    return x


def masked_mean(tensor, mask, dim=-1):
    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor.masked_fill_(~mask, 0.)

    total_el = mask.sum(dim=dim)
    mean = tensor.sum(dim=dim) / total_el.clamp(min=1.)
    mean.masked_fill_(total_el == 0, 0.)
    return mean


def cycle(iterable: Iterable):
    while True:
        for x in iterable:
            yield x


if __name__ == "__main__":
    x, y, z = torch.tensor(0.14159), torch.tensor(-1.0), torch.tensor(2.4)
    R = euler_to_rmat(x, y, z)
    print(x, y, z)
    print(*rmat_to_euler(R))
    vals = torch.randn((3, 4, 6))
    mats = six2rmat(vals)
    valback = rmat2six(mats)
    m1 = mats[:, :2]
    m2 = mats[:, 2:]
    res = rmat_cosine_dist(m1, m2)
    res2 = rmat_cosine_dist(m1, m1)
    res3 = rmat_cosine_dist(m2, m2)
    weight = 0.2
    out = so3_lerp(m1[0, 0], m2[0, 0], weight)
    log_m1 = log_rmat(m1[0, 0])
    log_rmat(torch.eye(3))

    rotvec = torch.tensor([[3.141592654, 0, 0]])
    log_mat = vec2skew(rotvec)
    rot_mat = torch.matrix_exp(log_mat)
    print(rot_mat)

    rot_pi = torch.tensor([[-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, 1.0],
                           ])

    log_r_pi = log_rmat(rot_pi)

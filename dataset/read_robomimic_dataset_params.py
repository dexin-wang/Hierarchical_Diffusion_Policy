import os
import sys
sys.path.append(os.curdir)
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R
import transforms3d as tfs
from hiera_diffusion_policy.model.common.rotation_transformer import RotationTransformer
import hiera_diffusion_policy.common.transform_utils as rtf


file = h5py.File('data/nonprehensile/tilt_fast.h5','r')
        
print('====== params | shape ======')
for i in file.keys():
    print('{:<10s} \t'.format(i), np.array(file[i]).shape)

print(file['data'].attrs["env_args"])

max_steps = 0
sum_steps = 0
demos = file['data']
for i in range(len(demos)):
    demo = demos[f'demo_{i}']
    max_steps = max(max_steps, demo['actions'][:].astype(np.float32).shape[0])
    sum_steps += demo['actions'][:].astype(np.float32).shape[0]
print('max_steps =', max_steps)
print('sum_steps =', sum_steps)


"""
rotation_transformer = RotationTransformer(
                from_rep='quaternion', to_rep='axis_angle')
# rotation_transformer = RotationTransformer(
#                 from_rep='axis_angle', to_rep='quaternion')

obs = file['data']['demo_0']['obs']
for key in obs.keys():
    a = np.array(obs[key]).astype(np.float32)
    print(key, a.shape)
actions = file['data']['demo_0']['actions'][:].astype(np.float32)
eef_pos = file['data']['demo_0']['obs']['robot0_eef_pos'][:].astype(np.float32)
eef_qua = file['data']['demo_0']['obs']['robot0_eef_quat'][:].astype(np.float32)

print('max(actions) =', np.max(actions)/20)

actions = actions[:100, :3]/20
eef_pos = eef_pos[:20]
eef_qua = eef_qua[:10]

eef_axisangle = [rtf.quat2axisangle(q) for q in eef_qua]
eef_axisangle = np.array(eef_axisangle)

# abs_actions_qua = rotation_transformer.forward(abs_actions[:, 3:])

print('demo length =\n', file['data']['demo_0']['actions'][:].astype(np.float32).shape[0])
print('actions =\n', actions)
print('eef_pos =\n', eef_pos)
print('eef_qua =\n', eef_qua)
print('eef_axisangle =\n', eef_axisangle)
# print('abs_actions_qua =\n', abs_actions_qua)

# print('scene_pcd.shape =', np.array(file['scene_pcd']).shape)
# print('object_pcd.shape =', np.array(file['object_pcd']).shape)
# print('goal.shape =', np.array(file['data']['demo_0']['goal']).shape)

# for i in range(eef_pos.shape[0]):
#     print('eef_pos =', eef_pos[i])
#     print('actions =', actions[i])

# print('====== params | num ======')
# print('{:<10s} \t'.format('episodes'), np.array(file['episode_ends']).shape[0])
# print('{:<10s} \t'.format('states'), np.array(file['state']).shape[0])
# print('{:<10s} \t'.format('object_pcd'), np.array(file['object_pcd']).shape[1])
# print('{:<10s} \t'.format('scene_pcd'), np.array(file['scene_pcd']).shape[1])
"""
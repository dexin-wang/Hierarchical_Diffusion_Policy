import numpy as np
import h5py
import json
from scipy.spatial.transform import Rotation as R
import transforms3d as tfs
import os
import sys
sys.path.append(os.curdir)
from hiera_diffusion_policy.model.common.rotation_transformer import RotationTransformer


file = h5py.File('data/nonprehensile/TriangularPrismLift_fast.h5','r')
        
print('====== params | shape ======')
for i in file.keys():
    print('{:<10s} \t'.format(i), np.array(file[i]).shape)

max_steps = 0
sum_steps = 0
demos = file['data']
for i in range(len(demos)):
    demo = demos[f'demo_{i}']
    max_steps = max(max_steps, demo['actions'][:].astype(np.float32).shape[0])
    sum_steps += demo['actions'][:].astype(np.float32).shape[0]
print('max_steps =', max_steps)
print('sum_steps =', sum_steps)

rotation_transformer = RotationTransformer(
                from_rep='quaternion', to_rep='axis_angle')
# rotation_transformer = RotationTransformer(
#                 from_rep='axis_angle', to_rep='quaternion')

obs = file['data']['demo_0']['obs']
for key in obs.keys():
    a = np.array(obs[key]).astype(np.float32)
    print(key, a.shape)
actions = file['data']['demo_0']['actions'][:].astype(np.float32)
eef_pos = file['data']['demo_0']['obs']['eef_pos'][:].astype(np.float32)
eef_qua = file['data']['demo_0']['obs']['eef_quat'][:].astype(np.float32)

eef_axisangle = rotation_transformer.forward(eef_qua)

abs_actions = actions[:10]
eef_pos = eef_pos[:10]
eef_qua = eef_qua[:10]
eef_axisangle = eef_axisangle[:10]

# abs_actions_qua = rotation_transformer.forward(abs_actions[:, 3:])

print('demo length =\n', file['data']['demo_0']['actions'][:].astype(np.float32).shape[0])
print('abs_actions =\n', abs_actions)
print('eef_pos =\n', eef_pos)
# print('eef_qua =\n', eef_qua)
# print('eef_ruler =\n', eef_euler)
print('eef_axisangle =\n', eef_axisangle)
# print('abs_actions_qua =\n', abs_actions_qua)

env_args = file["data"].attrs["env_args"]
env_meta = json.loads(file["data"].attrs["env_args"])
print('env_meta =', env_meta)
print('env_meta =', type(env_meta))
file.close()
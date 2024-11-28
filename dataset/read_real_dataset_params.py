import os
import sys
sys.path.append(os.curdir)
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R
import transforms3d as tfs
from hiera_diffusion_policy.model.common.rotation_transformer import RotationTransformer
import hiera_diffusion_policy.common.transform_utils as rtf


file = h5py.File('/home/wdx/research/diffusion_robot_manipulation/real_dataset/moveT_v2_repack.h5','r')
        
print('====== params | shape ======')
for k in file.keys():
    print('{:<10s} \t'.format(k), np.array(file[k]).shape)

for k in file['data']['demo_0'].keys():
    print('demo_0/{:<10s} \t'.format(k), np.array(file['data']['demo_0'][k]).shape)

for k in file['data']['demo_0']['obs'].keys():
    print('demo_0/obs/{:<10s} \t'.format(k), np.array(file['data']['demo_0']['obs'][k]).shape)

# actions = file['data']['demo_0']['actions'][:].astype(np.float32)
# absactions = file['data']['demo_0']['absaction'][:].astype(np.float32)

max_steps = 0
sum_steps = 0
demos = file['data']
for i in range(len(demos)):
    if f'demo_{i}' not in demos:
        print(f'demo_{i}'+'not exist')
        continue
    demo = demos[f'demo_{i}']
    max_steps = max(max_steps, demo['actions'][:].astype(np.float32).shape[0])
    sum_steps += demo['actions'][:].astype(np.float32).shape[0]
print('max_steps =', max_steps)
print('sum_steps =', sum_steps)


file.close()


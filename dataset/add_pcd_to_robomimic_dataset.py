"""
add scene_pcd, object_pcd and object_goal_pose to robomimic dataset, every demo
"""
import h5py
import sys
import os
sys.path.append(os.curdir)

import shutil
from hiera_diffusion_policy.common.robot import get_scene_object_pcd_goal
from tqdm import tqdm
import numpy as np


# dataset_path = 'data/low_dim_abs.hdf5'

dataset_path = sys.argv[1]

# create dataset file
newdataset_path = dataset_path.replace('abs.hdf5', 'abs_pcd.hdf5')
print(f'copying {dataset_path} to {newdataset_path}')
if os.path.exists(newdataset_path):
    os.remove(newdataset_path)
    print('Overwrite existing files')
shutil.copyfile(dataset_path, newdataset_path)

# get pcd and goal info
scene_pcd, object_pcd, object_goal_pose = get_scene_object_pcd_goal(newdataset_path, visual=False)

# read file and write new data
with h5py.File(newdataset_path, 'r+') as file:
    demos = file['data']
    for i in tqdm(range(len(demos)), desc="Write goal to h5py"):
        file['data'][f'demo_{i}']['goal'] = object_goal_pose
        file['data'][f'demo_{i}']['scene_pcd'] = scene_pcd
        file['data'][f'demo_{i}']['object_pcd'] = object_pcd

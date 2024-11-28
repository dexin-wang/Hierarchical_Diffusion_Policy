from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property


class ReplayBuffer:
    """
    {
        'data': {
            'state': None,  # gripper pose, finger position
            'action': None  # pos 6d-rote gripper
        },
        'meta': {
            'episode_ends': np.zeros((0,), dtype=np.int64),
            'goal': None,  # object goal position and quat
            'object_pcd': None,
            'scene_pcd': None,
        }
    }
    """
    def __init__(self, root: Dict[str,dict]):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        episode_ends为长度，索引为episode_ends-1
        """
        assert('data' in root)
        assert('meta' in root)
        assert('episode_ends' in root['meta'])
        for key, value in root['data'].items():
            assert(value.shape[0] == root['meta']['episode_ends'][-1])
        self.root = root
    
    @classmethod
    def create_empty_numpy(cls):
        root = {
            'data': {
            },
            'meta': {
                'episode_ends': np.zeros((0,), dtype=np.int64),
            }
        }
        return cls(root=root)
    
    # ============= properties =================
    @cached_property
    def data(self):
        return self.root['data']
    
    @cached_property
    def meta(self):
        return self.root['meta']

    @property
    def episode_ends(self):
        return self.meta['episode_ends']
    
    def get_episode_idxs(self):
        import numba
        numba.jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result
        return _get_episode_idxs(self.episode_ends)
        
    
    @property
    def backend(self):
        backend = 'numpy'
        if isinstance(self.root, zarr.Group):
            backend = 'zarr'
        return backend
    
    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(cls, src_store, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Load to memory.
        """
        src_root = zarr.group(src_store)
        root = None
        if store is None:
            # numpy backend
            meta = dict()
            for key, value in src_root['meta'].items():
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                keys = src_root['data'].keys()
            data = dict()
            for key in keys:
                arr = src_root['data'][key]
                data[key] = arr[:]

            root = {
                'meta': meta,
                'data': data
            }
        else:
            root = zarr.group(store=store)
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(source=src_store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
            data_group = root.create_group('data', overwrite=True)
            if keys is None:
                keys = src_root['data'].keys()
            for key in keys:
                value = src_root['data'][key]
                cks = cls._resolve_array_chunks(
                    chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value)
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store, dest=store,
                        source_path=this_path, dest_path=this_path,
                        if_exists=if_exists
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
        # print('root =', root['data']['keypoint'].shape) # (25650, 9, 2)
        # print('root =', root['data']['state'].shape)    # (25650, 5)
        # print('root =', root['data']['action'].shape)   # (25650, 2)
        buffer = cls(root=root)
        return buffer
    
    @classmethod
    def copy_from_path(cls, zarr_path, backend=None, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """
        if backend == 'numpy':
            print('backend argument is deprecated!')
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), 'r')
        return cls.copy_from_store(src_store=group.store, store=store, 
            keys=keys, chunks=chunks, compressors=compressors, 
            if_exists=if_exists, **kwargs)
    
    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == 'zarr':
            return str(self.root.tree())
        else:
            return super().__repr__()

    def data_keys(self):
        return self.data.keys()

    def meta_keys(self):
        return self.meta.keys()

    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]
    
    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def chunk_size(self):
        if self.backend == 'zarr':
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def add_pcd(self, scene_pcd, object_pcd):
        """
        scene_pcd: (N, 3)
        object_pcd: (N, 3)
        """
        self.meta['scene_pcd'] = scene_pcd
        self.meta['object_pcd'] = object_pcd

    """
    *合并后的 self.root:
            meta:
                episode_ends: (N,) N为轨迹数量
                scene_pcd: (N, 3)
                object_pcd: (N, 3)
            data:
                state: (ALL, S) ALL为所有轨迹包含的所有时刻的数量，包括 物体位姿7/末端位姿7/手指位置6
                action: (ALL, A)
                next_state: (ALL, S)
                obj_subgoal_world: (ALL, 7) obj_pos/obj_qua (world坐标系下)
                obj_subgoal_obspcd: (ALL, 7) obj_pos/obj_qua (观测点云坐标系下)
                fin_subgoal_world: (ALL, 8) lf_pos/rf_pos
    """

    def add_episode(self, 
                    data: Dict[str, np.ndarray],
                    subgoal=None,
                    scene_pcd=None, 
                    object_pcd=None):
        """
        - data: Dict{state; action, next_state, obj_subgoal_world; obj_subgoal_obspcd; fin_subgoal_world}
        - scene_pcd: (N,3)
        - object_pcd: (N,3)
        """
        assert(len(data) > 0)

        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            assert(len(value.shape) >= 1)
            if episode_length is None:
                episode_length = len(value)
            else:
                assert(episode_length == len(value))
        new_len = curr_len + episode_length

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            # create array
            if key not in self.data:
                # copy data to prevent modify
                arr = np.zeros(shape=new_shape, dtype=value.dtype)
                self.data[key] = arr
            else:
                arr = self.data[key]
                assert(value.shape[1:] == arr.shape[1:])
                # same method for both zarr and numpy
                arr.resize(new_shape, refcheck=False)
            # copy data
            arr[-value.shape[0]:] = value
        
        # append to episode ends
        episode_ends = self.episode_ends
        episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len
        
        if subgoal is not None:
            if 'subgoal' not in self.meta:
                self.meta['subgoal'] = np.zeros(shape=(1,)+subgoal.shape, dtype=subgoal.dtype)
                self.meta['subgoal'][0] = subgoal
            else:
                ori_episodes = self.meta['subgoal'].shape[0]
                arr = self.meta['subgoal']
                arr.resize((ori_episodes+1,)+subgoal.shape, refcheck=False)
                arr[-1] = subgoal
        
        if scene_pcd is not None:
            # add scene_pcd
            if 'scene_pcd' not in self.meta:
                self.meta['scene_pcd'] = np.zeros(shape=(1,)+scene_pcd.shape, dtype=scene_pcd.dtype)
                self.meta['scene_pcd'][0] = scene_pcd
            else:
                ori_episodes = self.meta['scene_pcd'].shape[0]
                arr = self.meta['scene_pcd']
                arr.resize((ori_episodes+1,)+scene_pcd.shape, refcheck=False)
                arr[-1] = scene_pcd
        
        if object_pcd is not None:
            # add object_pcd
            if 'object_pcd' not in self.meta:
                self.meta['object_pcd'] = np.zeros(shape=(1,)+object_pcd.shape, dtype=object_pcd.dtype)
                self.meta['object_pcd'][0] = object_pcd
            else:
                ori_episodes = self.meta['object_pcd'].shape[0]
                arr = self.meta['object_pcd']
                arr.resize((ori_episodes+1,)+object_pcd.shape, refcheck=False)
                arr[-1] = object_pcd

    
    def drop_episode(self):
        is_zarr = (self.backend == 'zarr')
        episode_ends = self.episode_ends[:].copy()
        assert(len(episode_ends) > 0)
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for key, value in self.data.items():
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends)-1)
        else:
            self.episode_ends.resize(len(episode_ends)-1, refcheck=False)
    
    def pop_episode(self):
        assert(self.n_episodes > 0)
        episode = self.get_episode(self.n_episodes-1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result
    
    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result
    
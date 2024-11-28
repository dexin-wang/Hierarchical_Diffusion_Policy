from typing import Optional
import numpy as np
import numba
from hiera_diffusion_policy.common.replay_buffer import ReplayBuffer


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask



@numba.jit(nopython=True)
def create_indices(
        episode_ends:np.ndarray, 
        sequence_length:int, 
        episode_mask: np.ndarray,
        pad_before: int=0, 
        pad_after: int=0) -> np.ndarray:
    """
    记录样本索引：list[[episode_idx, episode_length, sample_start_idx, sample_end_idx], ...]
        - 样本：包含sequence_length的action
        - episode_idx: 样本所在的episode索引
        - episode_length
        - sample_start_idx: 样本在episode中的起始索引，最小值为-pad_before，最大值为(episode_length + pad_after - sequence_length)
        - sample_end_idx:   样本在episode中的结束索引，等于 (start_idx+sequence_length-1)
    """
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        episode_start_idx = 0
        if i > 0:
            episode_start_idx = episode_ends[i-1]
        episode_end_idx = episode_ends[i]-1
        episode_length = episode_end_idx - episode_start_idx + 1
        
        # start_idx在[min, max]之间时，action序列前后需要填充pad的数量才不会超过设定值
        min_start_idx = -pad_before
        max_start_idx = episode_length + pad_after - sequence_length
        for idx in range(min_start_idx, max_start_idx+1):
            sample_start_idx = idx
            sample_end_idx = sample_start_idx + sequence_length - 1
            indices.append([i, episode_length, sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        abs_action: bool,
        sequence_length:int,    # 16
        pad_before:int=0,   # 7
        pad_after:int=0,    # 1
        episode_mask: Optional[np.ndarray]=None,
        ):
        assert(sequence_length >= 1)

        episode_ends = replay_buffer.episode_ends
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        self.indices = create_indices(
            episode_ends, 
            sequence_length=sequence_length, 
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=episode_mask
            )

        self.data_keys = list(replay_buffer.data_keys())
        self.meta_keys = list(replay_buffer.meta_keys())
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.abs_action = abs_action
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        """
        获取输入网络的原始数据
        
        return:
            - result: dict(
                meta: dict(
                    episode_ends: (1,)
                    scene_pcd: (n, 3)
                    object_pcd: (n, 3)
                    goal: (7,)
                    )
                data: dict(
                    state: (sequence_length, c)
                    subgoal: (sequence_length, c)
                    action: (sequence_length, c)
                    )
                )
        """
        episode_idx, episode_length, start_idx, end_idx = self.indices[idx]
        result = {
            'meta': dict(),
            'data': dict(),
        }
        for key in self.meta_keys:  # episode_ends scene_pcd object_pcd goal
            result['meta'][key] = np.array(self.replay_buffer.meta[key][episode_idx])
            
        for key in self.data_keys:
            if episode_idx == 0:
                episode_start_idx = 0
            else:
                episode_start_idx = self.replay_buffer.meta['episode_ends'][episode_idx-1]

            # state / action
            data = self.replay_buffer.data[key][episode_start_idx:
                                                self.replay_buffer.meta['episode_ends'][episode_idx]]
            # 获取action索引范围：start_idx -> 0 -> episode_length-1 -> end_idx
            pad_before_num = max(0-start_idx, 0)
            sample_start_idx = max(start_idx, 0)
            pad_after_num = max(end_idx - (episode_length-1), 0)
            sample_after_idx = min(end_idx, episode_length-1)
            # 原始数据
            sample = data[sample_start_idx: sample_after_idx+1]
            # pad before
            if pad_before_num > 0:
                pad_before = np.zeros((pad_before_num,)+sample.shape[1:], dtype=sample.dtype)
                # 只有非abs_action不进行替换
                if not (not self.abs_action and key == 'action'):
                    pad_before[:] = sample[0]
                sample = np.concatenate((pad_before, sample), axis=0)
            # pad after
            if pad_after_num > 0:
                pad_after = np.zeros((pad_after_num,)+sample.shape[1:], dtype=sample.dtype)
                # 只有非abs_action不进行替换
                if not (not self.abs_action and key == 'action'):
                    pad_after[:] = sample[-1]
                sample = np.concatenate((sample, pad_after), axis=0)
            assert sample.shape[0] == self.sequence_length
            result['data'][key] = sample

        return result
    
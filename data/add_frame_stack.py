import numpy as np
import os

from einops import rearrange

dataset_dir = f'{os.getcwd()}/data/data_img_obs_res_224'

train_img_obs = np.load(f'{dataset_dir}/train/traj_img_features.npy')
val_img_obs = np.load(f'{dataset_dir}/val/traj_img_features.npy')
test_img_obs = np.load(f'{dataset_dir}/test/traj_img_features.npy')


def stack_obs_all_trajectory(obs, num_frames=3):
    # Stack the observations
    stacked_img_obs = []
    for i in range(len(obs)):
        curr_stack_obs = [obs[i][0] for _ in range(num_frames)]
        curr_traj = []
        for j in range(len(obs[i])):
            curr_stack_obs.pop(0)
            curr_stack_obs.append(obs[i][j])
            curr_traj.append(np.concatenate(curr_stack_obs, axis=-1))
        stacked_img_obs.append(np.array(curr_traj))
    return np.array(stacked_img_obs)

n_frames = 3
train_stack_img_obs = stack_obs_all_trajectory(train_img_obs, num_frames=n_frames)
val_stack_img_obs = stack_obs_all_trajectory(val_img_obs, num_frames=n_frames)
test_stack_img_obs = stack_obs_all_trajectory(test_img_obs, num_frames=n_frames)

assert train_stack_img_obs.shape == (*train_img_obs.shape[:-1], train_img_obs.shape[-1]*n_frames), train_stack_img_obs.shape
assert val_stack_img_obs.shape == (*val_img_obs.shape[:-1], val_img_obs.shape[-1]*n_frames)
assert test_stack_img_obs.shape == (*test_img_obs.shape[:-1], test_img_obs.shape[-1]*n_frames)

np.save(f'{dataset_dir}/train/traj_img_features_stack.npy', train_stack_img_obs)
np.save(f'{dataset_dir}/val/traj_img_features_stack.npy', val_stack_img_obs)
np.save(f'{dataset_dir}/test/traj_img_features_stack.npy', test_stack_img_obs)
import numpy as np
import os

train_img_obs = np.load(f'{os.getcwd()}/data/data_img_obs_2/train/traj_img_obs.npy')
val_img_obs = np.load(f'{os.getcwd()}/data/data_img_obs_2/val/traj_img_obs.npy')
test_img_obs = np.load(f'{os.getcwd()}/data/data_img_obs_2/test/traj_img_obs.npy')


def stack_obs_all_trajectory(img_obs, num_frames=3):
    # Stack the observations
    stacked_img_obs = []
    for i in range(len(img_obs)):
        curr_stack_obs = [img_obs[i][0] for _ in range(num_frames)]
        curr_traj = []
        for j in range(len(img_obs[i])):
            curr_stack_obs.pop(0)
            curr_stack_obs.append(img_obs[i][j])
            curr_traj.append(np.concatenate(curr_stack_obs, axis=-1))
        stacked_img_obs.append(np.array(curr_traj))
    return np.array(stacked_img_obs)

num_frames = 3
train_stack_img_obs = stack_obs_all_trajectory(train_img_obs, num_frames=3)
val_stack_img_obs = stack_obs_all_trajectory(val_img_obs, num_frames=3)
test_stack_img_obs = stack_obs_all_trajectory(test_img_obs, num_frames=3)

assert train_stack_img_obs.shape == (*train_img_obs.shape[:-1], train_img_obs.shape[-1]*num_frames)
assert val_stack_img_obs.shape == (*val_img_obs.shape[:-1], val_img_obs.shape[-1]*num_frames)
assert test_stack_img_obs.shape == (*test_img_obs.shape[:-1], test_img_obs.shape[-1]*num_frames)

np.save(f'{os.getcwd()}/data/data_img_obs_2/train/traj_img_obs_stack.npy', train_stack_img_obs)
np.save(f'{os.getcwd()}/data/data_img_obs_2/val/traj_img_obs_stack.npy', val_stack_img_obs)
np.save(f'{os.getcwd()}/data/data_img_obs_2/test/traj_img_obs_stack.npy', test_stack_img_obs)
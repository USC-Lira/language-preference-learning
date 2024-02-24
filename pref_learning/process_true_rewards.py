import os
import json

import numpy as np


def process_data(args):
    # First concatenate the observations and actions to form the trajectory
    traj_actions = np.load(os.path.join(args.data_dir, 'with_camera_obs/traj_actions.npy'))
    traj_observations = np.load(os.path.join(args.data_dir, 'with_camera_obs/traj_observations.npy'))
    traj = np.concatenate((traj_observations, traj_actions), axis=-1)
    np.save(os.path.join(args.data_dir, 'traj.npy'), traj)

    # Then load the weight of reward and save it as a numpy array
    with open(os.path.join(args.data_dir, 'variant.json'), 'r') as f:
        configs = json.load(f)
        true_rewards = np.array(configs['eval_environment_kwargs']['weights'])

    np.save(os.path.join(args.data_dir, 'true_rewards.npy'), true_rewards)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='true_rewards/0')
    args = parser.parse_args()
    process_data(args)
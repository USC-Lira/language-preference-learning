import torch
from torch.utils.data import Dataset
import json
import numpy as np
import argparse
import os
from utils import generate_synthetic_comparisons_commands, generate_noisyaugmented_synthetic_comparisons_commands, \
    calc_and_set_global_vars


def get_comparisons(traj_i, traj_j, noise_augmentation=0, aug_comps=None, validation=False, split='train'):
    if noise_augmentation == 0:
        comps = generate_synthetic_comparisons_commands(traj_i, traj_j, augmented_comps=aug_comps,
                                                        validation=validation, split=split)

    else:
        comps = generate_noisyaugmented_synthetic_comparisons_commands(traj_i, traj_j, n_duplicates=noise_augmentation,
                                                                       augmented_comps=aug_comps, validation=validation,
                                                                       split=split)

    return comps


def generate_dataset(trajs, noise_augmentation=0, id_mapping=False, all_pairs=True, dataset_size=0, lang_aug=False,
                     validation=False, split='train'):
    dataset_traj_as = []
    dataset_traj_bs = []
    dataset_comps = []
    num_trajectories = len(trajs)

    # Prep work for noisy data augmentation
    if noise_augmentation:
        calc_and_set_global_vars(trajs)

    augmented_comps_mapping = None
    if lang_aug:
        augmented_comps_file = 'data/GPT_augmented_comps.json'
        with open(augmented_comps_file, 'r') as f:
            augmented_data = json.load(f)

        augmented_comps_mapping = {}
        for k in range(len(augmented_data)):
            augmented_comps_mapping[augmented_data[k][0]] = augmented_data[k]

    if all_pairs:
        print("GENERATING USING ALL-PAIRS METHOD.")
        for i in range(0, num_trajectories):
            print("GENERATING COMPARISONS FOR i =", i)
            for j in range(i + 1, num_trajectories):
                traj_i = trajs[i]
                traj_j = trajs[j]

                comps = get_comparisons(traj_i, traj_j, noise_augmentation=noise_augmentation,
                                        aug_comps=augmented_comps_mapping, validation=validation, split=split)
                flipped_comps = get_comparisons(traj_j, traj_i, noise_augmentation=noise_augmentation,
                                                aug_comps=augmented_comps_mapping, validation=validation, split=split)

                if id_mapping:  # With this option, we store the indexes of the `trajs` array rather than the actual trajectory
                    for c in comps:
                        dataset_traj_as.append(i)
                        dataset_traj_bs.append(j)
                        dataset_comps.append(c)
                    for fc in flipped_comps:
                        dataset_traj_as.append(j)
                        dataset_traj_bs.append(i)
                        dataset_comps.append(fc)
                else:
                    for c in comps:
                        dataset_traj_as.append(traj_i)
                        dataset_traj_bs.append(traj_j)
                        dataset_comps.append(c)
                    for fc in flipped_comps:
                        dataset_traj_as.append(traj_j)
                        dataset_traj_bs.append(traj_i)
                        dataset_comps.append(fc)

    else:
        print("GENERATING " + str(dataset_size) + " RANDOM COMPARISONS.")
        for n in range(dataset_size):
            print("GENERATING COMPARISONS FOR n =", n)
            i = 0
            j = 0
            while i == j:
                i = np.random.randint(num_trajectories)
                j = np.random.randint(num_trajectories)

            traj_i = trajs[i]
            traj_j = trajs[j]

            comps = get_comparisons(traj_i, traj_j, noise_augmentation=noise_augmentation)
            flipped_comps = get_comparisons(traj_j, traj_i, noise_augmentation=noise_augmentation)

            if id_mapping:  # With this option, we store the indexes of the `trajs` array rather than the actual trajectory
                for c in comps:
                    dataset_traj_as.append(i)
                    dataset_traj_bs.append(j)
                    dataset_comps.append(c)
                for fc in flipped_comps:
                    dataset_traj_as.append(j)
                    dataset_traj_bs.append(i)
                    dataset_comps.append(fc)
            else:
                for c in comps:
                    dataset_traj_as.append(traj_i)
                    dataset_traj_bs.append(traj_j)
                    dataset_comps.append(c)
                for fc in flipped_comps:
                    dataset_traj_as.append(traj_j)
                    dataset_traj_bs.append(traj_i)
                    dataset_comps.append(fc)

    return dataset_traj_as, dataset_traj_bs, dataset_comps


def generate_and_save_dataset(trajs, output_dir, noise_augmentation=0, id_mapping=False, all_pairs=True,
                              dataset_size=0, lang_aug=False, split='train'):
    traj_as, traj_bs, comps = generate_dataset(trajs, noise_augmentation=noise_augmentation, id_mapping=id_mapping,
                                               all_pairs=all_pairs, dataset_size=dataset_size, lang_aug=lang_aug,
                                               split=split)
    unique_comps = list(sorted(set(comps)))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print("SAVING TO:", output_dir)

    if id_mapping:
        # Change data type to int32 to save space
        traj_as = np.asarray(traj_as, dtype=np.int32)
        traj_bs = np.asarray(traj_bs, dtype=np.int32)
        np.save(os.path.join(output_dir, 'traj_a_indexes.npy'), traj_as)
        np.save(os.path.join(output_dir, 'traj_b_indexes.npy'), traj_bs)
    else:
        np.save(os.path.join(output_dir, 'traj_as.npy'), traj_as)
        np.save(os.path.join(output_dir, 'traj_bs.npy'), traj_bs)

    with open(os.path.join(output_dir, 'nlcomps.json'), 'w') as f:
        json.dump(comps, f)
    with open(os.path.join(output_dir, 'unique_nlcomps.json'), 'w') as f:
        json.dump(unique_comps, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--policy-dir', type=str, default='', help='')
    parser.add_argument('--output-dir', type=str, default='', help='')
    parser.add_argument('--dataset-size', type=int, default=1000, help='')
    parser.add_argument('--noise-augmentation', type=int, default=0, help='')
    parser.add_argument('--id-mapping', action="store_true", help='')
    parser.add_argument('--all-pairs', action="store_true", help='')
    parser.add_argument('--trajs-per-policy', type=int, default=5, help='')
    parser.add_argument('--trajs-per-expert-policy', type=int, default=5, help='')
    parser.add_argument('--val-split', type=float, default=0.1, help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--with-videos', action="store_true", help='')

    args = parser.parse_args()

    policy_dir = args.policy_dir
    output_dir = args.output_dir
    noise_augmentation = args.noise_augmentation
    id_mapping = args.id_mapping
    dataset_size = args.dataset_size
    all_pairs = args.all_pairs
    trajs_per_policy = args.trajs_per_policy
    trajs_per_expert_policy = args.trajs_per_expert_policy
    val_split = args.val_split
    seed = args.seed
    with_videos = args.with_videos

    np.random.seed(seed)

    # print("GETTING TRAJECTORY ROLLOUTS...")
    # trajectories = []
    # # trajectory_rewards = []
    # trajectory_video_ids = []
    # has_video_ids = True
    # for config in os.listdir(policy_dir):
    #     if with_videos:
    #         policy_path = os.path.join(policy_dir, config, "with_camera_obs")
    #     else:
    #         policy_path = os.path.join(policy_dir, config)
    #     if os.path.isdir(policy_path) and os.listdir(
    #             policy_path):  # Check that policy_path is a directory and that directory is not empty
    #         print(policy_path)
    #         observations = np.load(os.path.join(policy_path, "traj_observations.npy"))
    #         actions = np.load(os.path.join(policy_path, "traj_actions.npy"))
    #         try:
    #             video_ids = np.load(os.path.join(policy_path, "traj_video_ids.npy"))
    #         except FileNotFoundError:
    #             has_video_ids = False
    #
    #         # WARNING: rewards here is the reward given by the custom weights (0, 1, 2) that we used to train the
    #         # diverse policies, NOT the true ground truth Lift environment reward.
    #         # rewards = np.load(os.path.join(policy_path, "traj_rewards.npy"))
    #
    #         # observations has dimensions (n_trajs, n_timesteps, obs_dimension)
    #         trajs = np.concatenate((observations, actions), axis=-1)
    #
    #         # Downsample
    #         if config == 'expert':
    #             trajs = trajs[0:trajs_per_expert_policy]
    #             if has_video_ids:
    #                 video_ids = video_ids[0:trajs_per_expert_policy]
    #             # rewards = rewards[0:trajs_per_expert_policy]
    #         else:
    #             trajs = trajs[0:trajs_per_policy]
    #             if has_video_ids:
    #                 video_ids = video_ids[0:trajs_per_policy]
    #             # rewards = rewards[0:trajs_per_policy]
    #
    #         # NOTE: We use extend rather than append because we don't want to add an
    #         # additional dimension across the policies.
    #         trajectories.extend(trajs)
    #         # trajectory_rewards.extend(rewards)
    #         if has_video_ids:
    #             trajectory_video_ids.extend(video_ids)
    #
    # trajectories = np.asarray(trajectories)
    # # trajectory_rewards = np.asarray(trajectory_rewards)
    # num_trajectories = trajectories.shape[0]
    #
    # # Shuffle
    # p = np.random.permutation(num_trajectories)
    # trajectories = trajectories[p]
    # # trajectory_rewards = trajectory_rewards[p]
    #
    # # Split
    # split_i = int(np.ceil(val_split*num_trajectories))
    # val_trajectories = trajectories[0:split_i]
    # # val_trajectory_rewards = trajectory_rewards[0:split_i]
    # train_trajectories = trajectories[split_i:]
    # # train_trajectory_rewards = trajectory_rewards[split_i:]
    #
    # if has_video_ids:
    #     trajectory_video_ids = np.asarray(trajectory_video_ids)
    #     trajectory_video_ids = trajectory_video_ids[p]
    #     val_trajectory_video_ids = trajectory_video_ids[0:split_i]
    #     train_trajectory_video_ids = trajectory_video_ids[split_i:]
    #
    # print("NUM_TRAJECTORIES:", num_trajectories)
    # print("NUM TRAIN TRAJECTORIES:", len(train_trajectories))
    # print("NUM VAL TRAJECTORIES:", len(val_trajectories))
    # print("COMPILING DATASET:")

    data_dir = '/home/resl/language-preference-learning/data/dataset'
    train_trajectories = np.load(os.path.join(data_dir, 'train/trajs.npy'))
    val_trajectories = np.load(os.path.join(data_dir, 'val/trajs.npy'))
    # Further split train into train and val, and let current val be test.
    test_trajectories = val_trajectories
    split_i = len(val_trajectories)
    train_trajectories, val_trajectories = train_trajectories[split_i:], train_trajectories[:split_i]

    np.save(os.path.join(output_dir, 'train/trajs.npy'), train_trajectories)
    np.save(os.path.join(output_dir, 'val/trajs.npy'), val_trajectories)
    np.save(os.path.join(output_dir, 'test/trajs.npy'), test_trajectories)

    generate_and_save_dataset(train_trajectories, os.path.join(output_dir, 'train'), split='train',
                              id_mapping=args.id_mapping, lang_aug=True)
    generate_and_save_dataset(val_trajectories, os.path.join(output_dir, 'val'), split='val',
                              id_mapping=args.id_mapping, lang_aug=True)
    generate_and_save_dataset(test_trajectories, os.path.join(output_dir, 'test'), split='test',
                              id_mapping=args.id_mapping, lang_aug=True)

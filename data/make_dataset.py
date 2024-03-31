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

    np.save(os.path.join(output_dir, 'trajs.npy'), trajs)

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

    parser.add_argument('--output-dir', type=str, default='', help='')
    parser.add_argument('--dataset-size', type=int, default=1000, help='')
    parser.add_argument('--noise-augmentation', type=int, default=0, help='')
    parser.add_argument('--id-mapping', action="store_true", help='')
    parser.add_argument('--all-pairs', action="store_true", help='')
    parser.add_argument('--val-split', type=float, default=0.1, help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--use-img-obs', action="store_true", help='')
    

    args = parser.parse_args()

    output_dir = args.output_dir
    noise_augmentation = args.noise_augmentation
    id_mapping = args.id_mapping
    dataset_size = args.dataset_size
    all_pairs = args.all_pairs
    val_split = args.val_split
    seed = args.seed
    use_img_obs = args.use_img_obs
    

    np.random.seed(seed)

    data_dir = '/scr/zyang966/dataset_img_obs_2'
    train_trajectories = np.load(os.path.join(data_dir, 'train/trajs.npy'))
    val_trajectories = np.load(os.path.join(data_dir, 'val/trajs.npy'))

    total_num_trajs = len(train_trajectories) + len(val_trajectories)

    if use_img_obs:
        train_traj_img_obs = np.load(os.path.join(data_dir, 'train/traj_img_observations.npy'))
        val_traj_img_obs = np.load(os.path.join(data_dir, 'val/traj_img_observations.npy'))
        train_actions = np.load(os.path.join(data_dir, 'train/actions.npy'))
        val_actions = np.load(os.path.join(data_dir, 'val/actions.npy'))

    # Further split train into train and val, and let current val be test.
    test_trajectories = val_trajectories
    if use_img_obs:
        test_traj_img_obs = val_traj_img_obs
        test_actions = val_actions

    split_i = int(val_split * total_num_trajs)
    train_trajectories, val_trajectories = train_trajectories[split_i:], train_trajectories[:split_i]
    print("Number of train trajectories:", len(train_trajectories))
    print("Number of val trajectories:", len(val_trajectories))
    print("Number of test trajectories:", len(test_trajectories))
    
    if use_img_obs:
        train_traj_img_obs, val_traj_img_obs = train_traj_img_obs[split_i:], train_traj_img_obs[:split_i]
        train_actions, val_actions = train_actions[split_i:], train_actions[:split_i]

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    generate_and_save_dataset(train_trajectories, os.path.join(output_dir, 'train'),
                              noise_augmentation=noise_augmentation,
                              split='train', id_mapping=args.id_mapping, lang_aug=True)

    generate_and_save_dataset(val_trajectories, os.path.join(output_dir, 'val'), split='val',
                              id_mapping=args.id_mapping, lang_aug=True)

    generate_and_save_dataset(test_trajectories, os.path.join(output_dir, 'test'), split='test',
                              id_mapping=args.id_mapping, lang_aug=True)
    
    if use_img_obs:
        # Save the image observations and actions
        np.save(os.path.join(output_dir, 'train/traj_img_obs.npy'), train_traj_img_obs)
        np.save(os.path.join(output_dir, 'val/traj_img_obs.npy'), val_traj_img_obs)
        np.save(os.path.join(output_dir, 'test/traj_img_obs.npy'), test_traj_img_obs)

        np.save(os.path.join(output_dir, 'train/actions.npy'), train_actions)
        np.save(os.path.join(output_dir, 'val/actions.npy'), val_actions)
        np.save(os.path.join(output_dir, 'test/actions.npy'), test_actions)

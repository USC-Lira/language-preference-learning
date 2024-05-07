import json
import pickle
import numpy as np
import argparse
import os


ori_lang_comparisons = {
    "success": {
        "greater": [
            "Pick the spoon more succesfully",
            "Be more successful at picking the spoon",
        ],
        "less": ["Be less adept at picking the spoon", "Pick the spoon worse"],
    },
    "speed": {
        "greater": ["Move more quickly", "Be faster"],
        "less": ["Move more slowly", "Be slower"],
    },
    "avoid": {
        "greater": ["Detour to reach the spoon", "Avoid the pan better"],
        "less": ["Directly reach the spoon", "Be more straightforward to the spoon"],
    },
}


feature_labels_dict = {
    "success": ["success", "fail"],
    "speed": ["speed_fast", "speed_medium"],
    "avoid": ["avoid", "no_avoid"],
}


def downsample(obs, frames_per_traj=32):
    """
    Downsample the image observation to the desired number of frames
    """
    num_frames = obs.shape[0]
    downsampled_obs = np.zeros(
        (
            frames_per_traj,
            *obs.shape[1:],
        )
    )
    for i in range(frames_per_traj):
        idx = int(np.floor(i * num_frames / frames_per_traj))
        downsampled_obs[i] = obs[idx]

    return downsampled_obs


def load_trajectory_data(data_dir, frames_per_traj=32):
    """
    Load the trajectory data
    """
    states = []
    actions = []
    img_observations = []

    with open(os.path.join(data_dir, "traj_labels.pkl"), "rb") as f:
        traj_labels = pickle.load(f)

    traj_root_dir = os.path.join(data_dir, "trajectory")
    for traj_dir in os.listdir(traj_root_dir):
        state = np.load(os.path.join(traj_root_dir, traj_dir, "trajs.npy"))
        action = np.load(os.path.join(traj_root_dir, traj_dir, "actions.npy"))
        img_observation = np.load(
            os.path.join(traj_root_dir, traj_dir, "traj_img_obs.npy")
        )

        img_observation = downsample(img_observation, frames_per_traj).astype(np.uint8)
        state = downsample(state, frames_per_traj)
        actions.append(action)

        states.append(state)
        img_observations.append(img_observation)

    states = np.array(states)
    img_observations = np.array(img_observations)

    return states, actions, img_observations, traj_labels


def generate_lang_comparisons(feature, relation, augmented_comps, split="train", sample_comps_train=4):
    """
    Generate the language comparisons for the given feature and relation

    Args:
        - feature: str, the feature of the comparison
        - relation: str, the relation of the comparison
        - augmented_comps: dict, the augmented comparisons
        - split: str, the split of the dataset

    Returns:
        - lang_comps: list, the language comparisons
    """
    lang_comps = []
    ori_nlcomps = ori_lang_comparisons[feature][relation]

    for ori_nlcomp in ori_nlcomps:
        total_comps = len(augmented_comps[ori_nlcomp])
        if split == "train":
            lang_comps.extend(ori_nlcomps)
            num_comps_train = int(total_comps * 0.7)
            comps_idx = np.random.choice(num_comps_train, sample_comps_train, replace=False)
            new_comps = [augmented_comps[ori_nlcomp][idx] for idx in comps_idx]
            lang_comps.extend(new_comps)
        elif split == "val":
            lang_comps.extend(
                augmented_comps[ori_nlcomp][
                    int(total_comps * 0.7) : int(total_comps * 0.8)
                ]
            )
        elif split == "test":
            lang_comps.extend(
                augmented_comps[ori_nlcomp][int(total_comps * 0.8) :]
            )
        else:
            raise ValueError("Invalid split")

    return lang_comps


def save_data(data, save_dir):
    """
    Save the data to the given directory

    Args:
        - data: dict, the data to be saved
        - save_dir: str, the directory to save the data
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        return

    np.save(os.path.join(save_dir, "trajs.npy"), data["states"])
    np.save(os.path.join(save_dir, "traj_img_obs.npy"), data["img_observations"])


def split_and_save_dataset(
    data, data_dir, train_idx, val_idx, test_idx, train_split=0.6, val_split=0.2
):
    """
    Split the dataset and save it

    Args:
        - data: dict, the dataset
        - split: str, the split of the dataset
        - data_dir: str, the root directory of the data
    """
    # print(f"Splitting the dataset..")

    # # Split into train, val and test
    # num_trajs = len(data["states"])
    # num_train, num_val = int(train_split * num_trajs), int(val_split * num_trajs)

    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}
    test_data = {k: v[test_idx] for k, v in data.items()}

    print(
        f"Num of train: {len(train_idx)}, Num of val: {len(val_idx)}, Num of test: {len(test_idx)}"
    )

    save_data(train_data, os.path.join(data_dir, "train"))
    save_data(val_data, os.path.join(data_dir, "val"))
    save_data(test_data, os.path.join(data_dir, "test"))


def filter_labels_by_indices(indices, traj_labels):
    filtered_dict = {key: [] for key in traj_labels.keys()}

    # Populate the new dictionary with indices that are in the specified split
    for key, value_list in traj_labels.items():
        filtered_dict[key] = [index for index in value_list if index in indices]

    return filtered_dict


def generate_comparisons_in_range(
    traj_labels, traj_idx, actions, augmented_comps, root_dir, split="train"
):
    """
    Generate the comparisons in the given range of the indexes

    Args:
        - traj_labels: dict, the labels of the trajectories
        - range_start: int, the start of the range
        - range_end: int, the end of the range
        - augmented_comps: dict, the augmented comparisons
        - split: str, the split of the dataset

    Returns:
        - traj_a_indexes: list, the indexes of the first trajectory
        - traj_b_indexes: list, the indexes of the second trajectory
    """
    # filter out the indexes that are not in the range
    filtered_traj_labels = filter_labels_by_indices(traj_idx, traj_labels)

    traj_a_indexes, traj_b_indexes = [], []
    all_lang_comps = []

    # features = ["success", "avoid"]
    features = []
    for feature in features:
        greater_traj_idx = filtered_traj_labels[feature_labels_dict[feature][0]]
        less_traj_idx = filtered_traj_labels[feature_labels_dict[feature][1]]

        for i in greater_traj_idx:
            for j in less_traj_idx:
                assert i in traj_labels[feature_labels_dict[feature][0]], f"Index {i} should be in the greater trajectory"
                assert j in traj_labels[feature_labels_dict[feature][1]], f"Index {j} should be in the less trajectory"
                # compare b to a
                greater_lang_comps = generate_lang_comparisons(
                    feature, "greater", augmented_comps, split
                )
                for lang_comp in greater_lang_comps:
                    traj_a_indexes.append(np.where(traj_idx == j)[0][0])
                    traj_b_indexes.append(np.where(traj_idx == i)[0][0])
                    all_lang_comps.append(lang_comp)

                # compare a to b
                less_lang_comps = generate_lang_comparisons(
                    feature, "less", augmented_comps, split
                )
                for lang_comp in less_lang_comps:
                    traj_a_indexes.append(np.where(traj_idx == i)[0][0])
                    traj_b_indexes.append(np.where(traj_idx == j)[0][0])
                    all_lang_comps.append(lang_comp)
    
    # Compare every two trajectories based on speed
    # import ipdb; ipdb.set_trace()
    for i in range(len(traj_idx)):
        for j in range(i+1, len(traj_idx)):
            speeds_i = np.linalg.norm([action[:3] for action in actions[i]], axis=-1)
            speeds_j = np.linalg.norm([action[:3] for action in actions[j]], axis=-1)

            avg_speed_i, avg_speed_j = np.mean(speeds_i), np.mean(speeds_j)

            greater_lang_comps = generate_lang_comparisons(
                "speed", "greater", augmented_comps, split
            )
            for lang_comp in greater_lang_comps:
                if avg_speed_i < avg_speed_j:
                    traj_a_indexes.append(i)
                    traj_b_indexes.append(j)
                else:
                    traj_a_indexes.append(j)
                    traj_b_indexes.append(i)

                all_lang_comps.append(lang_comp)

            less_lang_comps = generate_lang_comparisons(
                "speed", "less", augmented_comps, split
            )
            for lang_comp in less_lang_comps:
                if avg_speed_i > avg_speed_j:
                    traj_a_indexes.append(i)
                    traj_b_indexes.append(j)
                else:
                    traj_a_indexes.append(j)
                    traj_b_indexes.append(i)

                all_lang_comps.append(lang_comp)

    traj_a_indexes = np.array(traj_a_indexes)
    traj_b_indexes = np.array(traj_b_indexes)

    save_dir = os.path.join(root_dir, split)
    np.save(os.path.join(save_dir, "traj_a_indexes.npy"), traj_a_indexes)
    np.save(os.path.join(save_dir, "traj_b_indexes.npy"), traj_b_indexes)
    with open(os.path.join(save_dir, "nlcomps.json"), "w") as f:
        json.dump(all_lang_comps, f)

    unique_nlcomps = list(set(all_lang_comps))
    with open(os.path.join(save_dir, "unique_nlcomps.json"), "w") as f:
        json.dump(unique_nlcomps, f)

    print(f"Split: {split}, Num of comparisons: {len(traj_a_indexes)}, Num of unique comparisons: {len(unique_nlcomps)}")


def generate_dataset(data_dir):
    """
    Generate the dataset for training, validation and testing

    Args:
        - trajs: list, a list of trajectories
        - trajs_label_dict: dict, a dictionary mapping labels to trajectories
        - noise_augmentation: int, the number of noise augmentation
        - lang_aug: bool, whether to perform language augmentation
        - split: str, the split of the dataset
    """

    # Load the data first
    states, actions, img_observations, traj_labels = load_trajectory_data(data_dir, frames_per_traj=40)
    data = {
        "states": states,
        "img_observations": img_observations,
    }

    with open(os.path.join(data_dir, "GPT_augmented_comparisons.json"), "r") as f:
        augmented_comps = json.load(f)

    train_traj_idx = np.load(os.path.join(data_dir, "train_indices.npy"))
    val_traj_idx = np.load(os.path.join(data_dir, "val_indices.npy"))
    test_traj_idx = np.load(os.path.join(data_dir, "test_indices.npy"))

    split_and_save_dataset(
        data,
        data_dir,
        train_idx=train_traj_idx,
        val_idx=val_traj_idx,
        test_idx=test_traj_idx,
    )

    # Generate the comparisons
    generate_comparisons_in_range(
        traj_labels, train_traj_idx, actions, augmented_comps, data_dir, "train"
    )

    generate_comparisons_in_range(
        traj_labels, val_traj_idx, actions, augmented_comps, data_dir, "val",
    )

    generate_comparisons_in_range(
        traj_labels, test_traj_idx, actions, augmented_comps, data_dir, "test",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    args = parser.parse_args()

    generate_dataset(args.data_dir)

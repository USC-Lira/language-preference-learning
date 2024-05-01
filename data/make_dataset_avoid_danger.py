import json
import pickle
import numpy as np
import argparse
import os

from data.transform import resize

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
        "greater": ["Detour to reach the spoon"],
        "less": ["Directly reach the spoon"],
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


def load_trajectory_data(data_root_dir, frames_per_traj=40):
    """
    Load the trajectory data
    """
    states = []
    actions = []
    img_observations = []

    with open(os.path.join(data_root_dir, "traj_labels.pkl"), "rb") as f:
        traj_labels = pickle.load(f)

    traj_root_dir = os.path.join(data_root_dir, "trajectory")
    for traj_dir in os.listdir(traj_root_dir):
        state = np.load(os.path.join(traj_root_dir, traj_dir, "trajs.npy"))
        action = np.load(os.path.join(traj_root_dir, traj_dir, "actions.npy"))
        img_observation = np.load(os.path.join(traj_root_dir, traj_dir, "traj_img_obs.npy"))

        img_observation = downsample(img_observation, frames_per_traj).astype(np.uint8)
        state = downsample(state, frames_per_traj)
        action = downsample(action, frames_per_traj)

        states.append(state)
        actions.append(action)
        img_observations.append(img_observation)

    states = np.array(states)
    actions = np.array(actions)
    img_observations = resize(img_observations, size=224)

    return states, actions, img_observations, traj_labels


def generate_lang_comparisons(feature, relation, augmented_comparisons, split="train"):
    """
    Generate the language comparisons for the given feature and relation

    Args:
        - feature: str, the feature of the comparison
        - relation: str, the relation of the comparison
        - augmented_comparisons: dict, the augmented comparisons
        - split: str, the split of the dataset

    Returns:
        - lang_comps: list, the language comparisons
    """
    lang_comps = []
    ori_nlcomps = ori_lang_comparisons[feature][relation]
    lang_comps.extend(ori_nlcomps)

    for ori_nlcomp in ori_nlcomps:
        if split == "train":
            lang_comps.extend(augmented_comparisons[ori_nlcomp][:15])
        elif split == "val":
            lang_comps.extend(augmented_comparisons[ori_nlcomp][15:17])
        elif split == "test":
            lang_comps.extend(augmented_comparisons[ori_nlcomp][17:])
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
    np.save(os.path.join(save_dir, "actions.npy"), data["actions"])
    np.save(os.path.join(save_dir, "traj_img_obs.npy"), data["img_observations"])


def split_and_save_dataset(data, data_root_dir, train_split=0.8, val_split=0.1):
    """
    Split the dataset and save it

    Args:
        - data: dict, the dataset
        - split: str, the split of the dataset
        - data_root_dir: str, the root directory of the data
    """
    print(f"Splitting the dataset..")

    # Split into train, val and test
    num_trajs = len(data["states"])
    num_train, num_val = int(train_split * num_trajs), int(val_split * num_trajs)

    train_data = {k: v[:num_train] for k, v in data.items()}
    val_data = {k: v[num_train : num_train + num_val] for k, v in data.items()}
    test_data = {k: v[num_train + num_val :] for k, v in data.items()}

    save_data(train_data, os.path.join(data_root_dir, "train"))
    save_data(val_data, os.path.join(data_root_dir, "val"))
    save_data(test_data, os.path.join(data_root_dir, "test"))

    return num_train, num_val


def generate_comparisons_in_range(
    traj_labels, range_start, range_end, augmented_comparisons, root_dir, split="train"
):
    """
    Generate the comparisons in the given range of the indexes

    Args:
        - traj_labels: dict, the labels of the trajectories
        - range_start: int, the start of the range
        - range_end: int, the end of the range
        - augmented_comparisons: dict, the augmented comparisons
        - split: str, the split of the dataset

    Returns:
        - traj_a_indexes: list, the indexes of the first trajectory
        - traj_b_indexes: list, the indexes of the second trajectory
    """
    # filter out the indexes that are not in the range
    filtered_traj_labels = {
        k: [i for i in v if range_start <= i <= range_end] for k, v in traj_labels.items()
    }

    traj_a_indexes, traj_b_indexes = [], []
    all_lang_comps = []

    features = ["success", "avoid", "speed"]
    for feature in features:
        greater_traj_idx = filtered_traj_labels[feature_labels_dict[feature][0]]
        less_traj_idx = filtered_traj_labels[feature_labels_dict[feature][1]]
        for i in greater_traj_idx:
            for j in less_traj_idx:
                # compare b to a
                greater_lang_comps = generate_lang_comparisons(
                    feature, "greater", augmented_comparisons, split
                )
                for lang_comp in greater_lang_comps:
                    traj_a_indexes.append(j)
                    traj_b_indexes.append(i)
                    all_lang_comps.append(lang_comp)

                # compare a to b
                less_lang_comps = generate_lang_comparisons(feature, "less", augmented_comparisons, split)
                for lang_comp in less_lang_comps:
                    traj_a_indexes.append(i)
                    traj_b_indexes.append(j)
                    all_lang_comps.append(lang_comp)

    traj_a_indexes = np.array(traj_a_indexes)
    traj_b_indexes = np.array(traj_b_indexes)

    save_dir = os.path.join(root_dir, split)
    np.save(os.path.join(save_dir, "traj_a_indexes.npy"), traj_a_indexes)
    np.save(os.path.join(save_dir, "traj_b_indexes.npy"), traj_b_indexes)
    with open(os.path.join(save_dir, "nlcomps.json"), "w") as f:
        json.dump(all_lang_comps, f)


def generate_dataset(data_root_dir):
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
    states, actions, img_observations, traj_labels = load_trajectory_data(data_root_dir)
    data = {
        "states": states,
        "actions": actions,
        "img_observations": img_observations,
    }

    with open(os.path.join(data_root_dir, "GPT_augmented_comparisons.json"), "r") as f:
        augmented_comparisons = json.load(f)

    num_train, num_val = split_and_save_dataset(data, data_root_dir)

    # Generate the comparisons
    generate_comparisons_in_range(
        traj_labels, 0, num_train - 1, augmented_comparisons, data_root_dir, "train"
    )

    generate_comparisons_in_range(
        traj_labels,
        num_train,
        num_train + num_val - 1,
        augmented_comparisons,
        data_root_dir,
        "val",
    )

    generate_comparisons_in_range(
        traj_labels,
        num_train + num_val,
        len(states) - 1,
        augmented_comparisons,
        data_root_dir,
        "test",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", type=str)
    args = parser.parse_args()

    generate_dataset(args.data_root_dir)
import numpy as np
import os
import re
import pickle
from PIL import Image
from collections import defaultdict


def read_single_traj(traj_dir):
    """
    Read a single trajectory from the dataset

    Args:
        - traj_dir: str, the directory of the trajectory

    Returns:
        - states: np.ndarray, the states of the trajectory
        - actions: np.ndarray, the actions of the trajectory
        - images: np.ndarray, the image observations of the trajectory
    """

    # First read the states and actions
    state_file = os.path.join(traj_dir, "obs_dict.pkl")

    with open(state_file, "rb") as f:
        obs_dict = pickle.load(f)
        states = np.concatenate(
            [obs_dict["state"], obs_dict["qpos"], obs_dict["qvel"]], axis=-1
        )

        # Augment state with position difference
        pos_diff = np.diff(states[:, :3], axis=0)
        pos_diff = np.concatenate([np.zeros((1, 3)), pos_diff], axis=0)
        states = np.concatenate([states, pos_diff], axis=-1)

        states = states[:-1, :]  # Remove the last state

    action_file = os.path.join(traj_dir, "policy_out.pkl")
    with open(action_file, "rb") as f:
        policy_out = pickle.load(f)
        actions = np.concatenate(
            [policy_out[i]["actions"][np.newaxis, :] for i in range(len(policy_out))],
            axis=0,
        )
    
    states = np.concatenate([states, actions], axis=-1)

    # Then read the images
    # First sort the images by timestep order
    img_dir = os.path.join(traj_dir, "images0")

    # Read all image files by timestep order
    image_files = os.listdir(img_dir)

    def extract_number(filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        return None
    
    sorted_img_files = sorted(image_files, key=extract_number)

    images = []
    for img_file in sorted_img_files:
        # load jpg image and convert to numpy array
        img = Image.open(os.path.join(img_dir, img_file))
        img = np.asarray(img)
        images.append(img[np.newaxis, :])

    images = np.concatenate(images, axis=0)
    images = images[:-1, :, :, :]  # Remove the last image

    return {"states": states, "actions": actions, "images": images}


def read_all_traj(data_root_dir, save_root_dir):
    """
    Read all trajectories from the dataset

    Args:
        - data_dir: str, the directory of the dataset

    Returns:
        - all_trajs: list, a list of trajectories
    """
    all_trajs = []
    traj_id = 0
    traj_labels_dict = defaultdict(list)

    for dirpath, dirnames, filenames in os.walk(data_root_dir):
        # Extract the path parts to form labels
        parts = dirpath.split(os.sep)
        if parts[-1].startswith("traj"):
            labels = parts[1:-2]  # Skip the root_dir and trajectory folder in the label

            for label in labels:
                traj_labels_dict[label].append(traj_id)
            
            traj_data = read_single_traj(dirpath)
            traj_save_dir = os.path.join(save_dir, f"trajectory/traj{traj_id}")
            os.makedirs(traj_save_dir, exist_ok=True)
            np.save(os.path.join(traj_save_dir, "trajs.npy"), traj_data["states"])
            np.save(os.path.join(traj_save_dir, "actions.npy"), traj_data["actions"])
            np.save(os.path.join(traj_save_dir, "traj_img_obs.npy"), traj_data["images"])

            traj_id += 1

            if traj_id % 10 == 0:
                print(f"Processed {traj_id} trajectories")

    # Save the labels
    with open(os.path.join(save_root_dir, "traj_labels.pkl"), "wb") as f:
        pickle.dump(traj_labels_dict, f)

    return all_trajs


if __name__ == "__main__":
    data_dir = "dataset_avoid_danger"
    save_dir = "data_avoid_danger"

    read_all_traj(data_dir, save_dir)
    

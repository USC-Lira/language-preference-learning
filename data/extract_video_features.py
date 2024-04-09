""" Extract video features from a video of trajecotories with S3D model. """
import os
import numpy as np
import torch
from einops import rearrange

from S3D_HowTo100M.s3dg import S3D


def downsample_frames(frames, target_frame_count=32):
    """
    Downsample a list of frames to a target number of frames.
    
    Parameters:
        - frames: A list of frames.
        - target_frame_count: The target number of frames to downsample to, including the first and last frames.
    
    Returns:
        - A list of downsampled frames.
    """
    total_frames = len(frames)
    n_step = (total_frames - 1) / (target_frame_count - 1)
    selected_frames = [frames[int(i * n_step)] for i in range(target_frame_count)]
    return selected_frames


def extract_features(model, img_obs):
    """
    Extract features from a list of video trajectories.

    Parameters:
        - model: The S3D-G model.
        - img_obs: A list of video trajectories.

    Returns:
        - A list of features extracted from the video trajectories.
    """
    features = []
    for i in range(len(img_obs)):
        curr_img_obs = img_obs[i]

        # Downsample to 32 frames, including the first and last frames
        curr_img_obs = downsample_frames(curr_img_obs)

        curr_img_obs = rearrange(curr_img_obs, 't h w c -> 1 c t h w')
        curr_img_obs = torch.from_numpy(curr_img_obs).float().to('cuda')

        # Get the features from the mixed_5c layer of S3D
        video_output = model(curr_img_obs)
        feature = video_output['mixed_5c']

        features.append(feature.squeeze(0).detach().cpu().numpy())
        
        if i % 10 == 0:
            print(f'Processed {i} trajectories')
    
    features = rearrange(features, 'b d -> b d')
    assert features.shape[0] == len(img_obs), f'{features.shape[0]} != {len(img_obs)}'

    return features


if __name__ == '__main__':
    # Instantiate the model
    net = S3D('s3d_dict.npy', 512)

    # Load the model weights
    net.load_state_dict(torch.load('s3d_howto100m.pth'))
    net.to('cuda')
    net.eval()

    # Load the data
    dataset_dir = f'{os.getcwd()}/data/data_img_obs_res_224'
    train_img_obs = np.load(f'{dataset_dir}/train/traj_img_obs.npy')
    val_img_obs = np.load(f'{dataset_dir}/val/traj_img_obs.npy')
    test_img_obs = np.load(f'{dataset_dir}/test/traj_img_obs.npy')

    train_img_features = extract_features(net, train_img_obs)
    np.save(f'{dataset_dir}/train/traj_img_features_s3d.npy', train_img_features)

    val_img_features = extract_features(net, val_img_obs)
    np.save(f'{dataset_dir}/val/traj_img_features_s3d.npy', val_img_features)

    test_img_features = extract_features(net, test_img_obs)
    np.save(f'{dataset_dir}/test/traj_img_features_s3d.npy', test_img_features)
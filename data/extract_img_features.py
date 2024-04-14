import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3
from torchvision.models import resnet18, ResNet18_Weights

from einops import rearrange

dataset_dir = dataset_dir = f'{os.getcwd()}/data/data_seg_img_obs_res_224'

train_img_obs = np.load(f'{dataset_dir}/train/traj_img_obs.npy')
val_img_obs = np.load(f'{dataset_dir}/val/traj_img_obs.npy')
test_img_obs = np.load(f'{dataset_dir}/test/traj_img_obs.npy')


# extractor_name = 'resnet18'
# net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to('cuda')
extractor_name = 'efficientnetb3'
net = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).to('cuda')
extractor = nn.Sequential(*list(net.children())[:-1])
transform = ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True)


def extract_features(img_obs, batch_size=32):
    features = []
    for i in range(len(img_obs)):
        curr_traj = []
        # import ipdb; ipdb.set_trace()
        for j in range(int(np.ceil(len(img_obs[i]) / batch_size))):
            x = torch.tensor(img_obs[i][j * batch_size: (j + 1) * batch_size])
            x = rearrange(x, 'b h w c -> b c h w')
            x = transform(x)
            feature = extractor(x.to('cuda'))
            feature = torch.flatten(feature, 1)
            curr_traj.append(feature.detach().cpu().numpy())
        
        curr_traj = np.concatenate(curr_traj, axis=0)
        assert curr_traj.shape[0] == len(img_obs[i]), f'{curr_traj.shape[0]} != {len(img_obs[i])}'

        features.append(curr_traj)
        
        if i % 10 == 0:
            print(f'Processed {i} trajectories')
    
    features = rearrange(features, 'b t d -> b t d')
    assert features.shape[0] == len(img_obs), f'{features.shape[0]} != {len(img_obs)}'

    return features

train_img_features = extract_features(train_img_obs)
np.save(f'{dataset_dir}/train/traj_img_features_{extractor_name}.npy', train_img_features)

val_img_features = extract_features(val_img_obs)
np.save(f'{dataset_dir}/val/traj_img_features_{extractor_name}.npy', val_img_features)

test_img_features = extract_features(test_img_obs)
np.save(f'{dataset_dir}/test/traj_img_features_{extractor_name}.npy', test_img_features)
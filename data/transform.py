import os
import argparse
import torch
import numpy as np
from einops import rearrange

from torchvision import transforms
from torchvision.utils import save_image


def resize(img_obs, size=112, flip=False):
    """
    Resize the images
    """
    img_obs = torch.tensor(img_obs).float() / 255.0
    total_num = img_obs.shape[0]

    img_obs = rearrange(img_obs, 'b t h w c -> (b t) c h w')

    resize = transforms.Compose([
        transforms.Resize((112, 112)),
    ]
    )

    if flip:
        resize = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.functional.vflip,
        ]
        )

    resized_img_obs = torch.zeros((img_obs.shape[0], 3, size, size))

    # Resize the images with batch
    batch_size = 100
    for i in range(img_obs.shape[0] // batch_size):
        if i % 10 == 0:
            print(f'{i * batch_size} Images processed...')
        resized_img_obs[i*batch_size:(i+1)*batch_size] = resize(img_obs[i*batch_size:(i+1)*batch_size])

    # save one image as an example
    save_image(resized_img_obs[0], 'resized_img_obs_example.png')

    resized_img_obs = rearrange(resized_img_obs, '(b t) c h w -> b t h w c', b=total_num)

    return resized_img_obs


def flip(img_obs):
    """
    Flip the images vertically
    """
    img_obs = torch.tensor(img_obs).float()
    if img_obs.max() > 1:
        img_obs /= 255.0
    num_trajs = img_obs.shape[0]
    
    img_obs = rearrange(img_obs, 'b t h w c -> (b t) c h w')

    flip = transforms.Compose([
        transforms.functional.vflip,
    ])

    flipped_img_obs = torch.zeros((img_obs.shape[0], 3, 224, 224))

    # Flip the images with batch
    batch_size = 100
    for i in range(img_obs.shape[0] // batch_size):
        if i % 10 == 0:
            print(f'{i * batch_size} Images processed...')
        flipped_img_obs[i*batch_size:(i+1)*batch_size] = flip(img_obs[i*batch_size:(i+1)*batch_size])

    # save one image as an example
    save_image(flipped_img_obs[0], 'flipped_img_obs_example.png')

    flipped_img_obs = rearrange(flipped_img_obs, '(b t) c h w -> b t h w c', b=num_trajs)

    return flipped_img_obs


def crop(img_obs):
    """
    Center crop the images
    """
    img_obs = torch.tensor(img_obs[:10]).float()
    if img_obs.max() > 1:
        img_obs = img_obs / 255.0
    num_trajs = img_obs.shape[0]
    
    img_obs = rearrange(img_obs, 'b t h w c -> (b t) c h w')

    # crop = transforms.Compose([
    #     transforms.functional.crop(32, 32, 160, 160),
    #     # transforms.CenterCrop(192),
    #     # transforms.RandomCrop(160),
    # ]
    # )

    cropped_img_obs = torch.zeros((img_obs.shape[0], 3, 172, 172))

    # Crop the images with batch
    batch_size = 100
    for i in range(img_obs.shape[0] // batch_size):
        if i % 10 == 0:
            print(f'{i * batch_size} Images processed...')
        cropped_img_obs[i*batch_size:(i+1)*batch_size] = transforms.functional.crop(img_obs[i*batch_size:(i+1)*batch_size],
                                                                                    15, 26,
                                                                                    172, 172)
    
    cropped_img_obs = transforms.Resize((224, 224))(cropped_img_obs)

    # save one image as an example
    save_image(cropped_img_obs[0], 'cropped_img_obs_example.png')

    cropped_img_obs = rearrange(cropped_img_obs, '(b t) c h w -> b t h w c', b=num_trajs)

    return cropped_img_obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', type=str, help='which transformation to apply', 
                        choices=['resize', 'flip', 'crop'], default='resize')
    parser.add_argument('--data-dir', type=str, default='data_seg_img_obs_res_224/train')
    args = parser.parse_args()

    # Load the images
    img_obs = np.load(f'{args.data_dir}/traj_img_obs.npy')
    
    transfrom_funcs = {
        'resize': resize,
        'flip': flip,
        'crop': crop,
    }

    # Apply the transformations
    transformed_img_obs = transfrom_funcs[args.transform](img_obs)

    # Save the resized images
    save_dir = args.data_dir
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/traj_img_obs.npy', transformed_img_obs.numpy())



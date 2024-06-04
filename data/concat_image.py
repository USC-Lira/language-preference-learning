import os
import argparse
import numpy as np

from einops import rearrange

def concat_image(data_dir):
    data = []
    for file in os.listdir(data_dir):
        if file.endswith('.npy'):
            data.append(np.load(os.path.join(data_dir, file)))

    data = rearrange(data, 'b t h w c -> b t h w c')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_seg_img_obs_res_224')
    args = parser.parse_args()

    data = concat_image(args.data_dir)

    print(data.shape)

    # Save the concatenated image
    np.save(f'{args.data_dir}/traj_img_obs.npy', data)
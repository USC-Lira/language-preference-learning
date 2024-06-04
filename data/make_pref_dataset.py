import os
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data-dir', type=str, default='', help='')
    parser.add_argument('--val-split', type=float, default=0.1, help='')
    parser.add_argument('--use-img-obs', action="store_true", help='')

    args = parser.parse_args()

    data_dir = args.data_dir
    use_img_obs = args.use_img_obs

    trajs = []
    trajs_img_obs = []
    for split in ['val', 'test']:
        trajs.append(np.load(os.path.join(data_dir, f"{split}/trajs.npy")))
        if use_img_obs:
            trajs_img_obs.append(np.load(os.path.join(data_dir, f"{split}/traj_img_obs.npy")))
    
    # concat trajs
    trajs = np.concatenate(trajs, axis=0)
    if use_img_obs:
        trajs_img_obs = np.concatenate(trajs_img_obs, axis=0)
    import ipdb; ipdb.set_trace()
    print("Loaded trajs")
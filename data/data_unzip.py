import pickle, bz2
import os
import numpy as np
import time

t0 = time.time()
# Load the data
train_img_obs = pickle.load(bz2.open('/scr/zyang966/data_img_obs/train/traj_img_observations.pkl.bz2', 'r'))
val_img_obs = pickle.load(bz2.open('/scr/zyang966/data_img_obs/train/traj_img_observations.pkl.bz2', 'r'))
print('Time to load data:', time.time() - t0)

t1 = time.time()
# Save in the format of the original data
np.save(os.path.join('/scr/zyang966/data_img_obs/train', 'traj_img_observations.npy'), train_img_obs)
np.save(os.path.join('/scr/zyang966/data_img_obs/val', 'traj_img_observations.npy'), val_img_obs)
print('Time to save data:', time.time() - t1)
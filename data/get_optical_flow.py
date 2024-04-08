import cv2
import os
import numpy as np

data_dir = 'data/data_img_obs_res_224'
img_observations = np.load(f'{data_dir}/train/traj_img_obs.npy')

test_img_obs = img_observations[0, ::10]

prvs = cv2.cvtColor(test_img_obs[0], cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(test_img_obs[0])
hsv[..., 1] = 255

i = 0

optical_flow_dir = f'{data_dir}/optical_flow'
os.makedirs(optical_flow_dir, exist_ok=True)

for i in range(1, len(test_img_obs)):
    next = cv2.cvtColor(test_img_obs[i], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if i % 10 == 0:
        print(f'Processed {i} images')
        cv2.imwrite(f'{data_dir}/optical_flow/opticalfb_{i}.png', next)
        cv2.imwrite(f'{data_dir}/optical_flow/opticalhsv_{i}.png', bgr)
    prvs = next

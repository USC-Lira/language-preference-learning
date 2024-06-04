import re
import os
import numpy as np
from PIL import Image

from lang_pref_learning.real_robot_exp.utils import replay_trajectory_video


images = []
img_dir = '/home/resl/Downloads/dataset_avoid_danger/no_pan/no_avoid/speed_fast/success/1/traj70/images0'

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

replay_trajectory_video(images, frame_rate=10)
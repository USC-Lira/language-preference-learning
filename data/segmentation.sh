source /scr/zyang966/miniconda3/etc/profile.d/conda.sh
conda activate detic

/scr/zyang966/conda_envs/detic/bin/python3 data/get_segmentation_mask.py \
--data-dir=data/data_img_obs_res_224_30k/test --start-index=18 \
--config-file Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
--vocabulary custom --custom_vocabulary bottle,gripper,cube --confidence-threshold 0.1 \
--opts MODEL.WEIGHTS Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \

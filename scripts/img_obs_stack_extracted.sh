python -m lang_pref_learning.feature_learning.learn_features \
--initial-loss-check \
--data-dir=data/data_img_obs_res_224 \
--exp-name=img-obs-seq-200-stack-extract-features \
--bert-model=bert-tiny --use-bert-encoder --traj-encoder=cnn \
--batch-size=512 --add-norm-loss --epochs=1 \
--use-img-obs \
# --use-stack-img-obs --n-frames=3
# --use-visual-features --visual-feature-dim=1536 --encoder-hidden-dim=256 --seq-len=500
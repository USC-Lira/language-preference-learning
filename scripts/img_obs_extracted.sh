python -m lang_pref_learning.feature_learning.learn_features \
--initial-loss-check \
--data-dir=data/data_seg_img_obs_res_224 \
--exp-name=resnet18-features-add-state \
--bert-model=bert-tiny --use-bert-encoder --traj-encoder=cnn \
--batch-size=512 --add-norm-loss --epochs=2 \
--use-img-obs --seq-len=500 --resample --resample-factor=0.1 \
--use-visual-features --visual-feature-dim=512 --feature-extractor=resnet18 \
--encoder-hidden-dim=256
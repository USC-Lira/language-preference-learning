python -m lang_pref_learning.feature_learning.learn_features \
--initial-loss-check \
--data-dir=data/data_img_obs_res_224_more \
--exp-name=img-obs-feature-s3d-more-data \
--bert-model=bert-tiny --use-bert-encoder --traj-encoder=cnn \
--batch-size=1024 --add-norm-loss --epochs=5 \
--use-img-obs \
--use-visual-features --visual-feature-dim=1024 --feature-extractor=s3d \
--encoder-hidden-dim=128 --lr=2e-3
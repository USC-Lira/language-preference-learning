python -m lang_pref_learning.feature_learning.learn_features \
--initial-loss-check \
--data-dir=data/data_img_obs_res_112_more \
--exp-name=raw-img-obs-cnn \
--bert-model=bert-tiny --use-bert-encoder \
--batch-size=64 --add-norm-loss --epochs=1 \
--use-img-obs --traj-encoder=cnn --seq-len=500 --resample --resample-factor=0.1

python -m lang_pref_learning.feature_learning.learn_features \
--initial-loss-check \
--data-dir=data/data_img_obs_res_224_more \
--exp-name=raw-img-cnn-t5-small \
--lang-model=t5-small --use-bert-encoder \
--batch-size=64 --add-norm-loss --epochs=1 \
--use-img-obs --traj-encoder=cnn --seq-len=500 --resample --resample-factor=0.1
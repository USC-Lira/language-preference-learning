python -m lang_pref_learning.feature_learning.learn_features \
--initial-loss-check \
--data-dir=data/data_img_obs_num_99 \
--exp-name=img-obs-stack-2-resample \
--lang-model=bert-tiny --use-bert-encoder \
--batch-size=64 --add-norm-loss --epochs=1 \
--use-img-obs --traj-encoder=cnn --use-stack-img-obs --n-frames=2 --seq-len=500 \
--resample --resample-factor=0.1

python -m lang_pref_learning.feature_learning.learn_features --initial-loss-check \
--env=robosuite --data-dir=data/data_img_obs_res_224_more \
--exp-name=robosuite-img-obs-t5-base \
--lang-model=t5-base --use-bert-encoder \
--batch-size=64 --add-norm-loss --epochs=1 \
--use-img-obs --traj-encoder=cnn --seq-len=500 --resample --resample-factor=0.1 \
--traj-reg-coeff=2e-2 --traj-reg-margin=1.0

python lang_pref_learning/real_robot_exp/pref_learning.py \
--env=widowx \
--data-dir=data/avoid_danger_user_study \
--model-dir=exp/widowx-img-obs_20240509_160535_lr_0.001_schedule_False \
--use-bert-encoder \
--lang-model=t5-small \
--use-img-obs \
--method=comp \
--traj-encoder=cnn
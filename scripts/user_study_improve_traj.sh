python lang_pref_learning/real_robot_exp/improve_trajectory.py \
--env=widowx \
--model-dir=exp/widowx-img-obs_20240509_160535_lr_0.001_schedule_False \
--data-dir=data/avoid_danger_user_study \
--use-lang-encoder --lang-model=t5-small \
--iterations=10 \
--use-image-obs \
--seed=0
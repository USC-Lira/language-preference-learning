python lang_pref_learning/model_analysis/improve_trajectory.py \
--env=metaworld \
--data-dir=data/metaworld \
--model-dir=exp/mw-t5-small_20240601_085453_lr_0.002_schedule_False \
--traj-encoder=mlp \
--lang-model=t5-small \
--use-bert-encoder \
--seed=42 \
--iterations=15 \
--num-trials=100
python -m lang_pref_learning.feature_learning.learn_features \
--initial-loss-check \
--data-dir=data/data_img_obs_2 \
--exp-name=finetune-bert-tiny-img-obs \
--bert-model=bert-tiny --use-bert-encoder \
--batch-size=64 --add-norm-loss --epochs=1 \
--use-img-obs --traj-encoder=cnn

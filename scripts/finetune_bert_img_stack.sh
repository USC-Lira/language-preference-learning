python -m lang_pref_learning.feature_learning.learn_features \
--initial-loss-check \
--data-dir=data/data_img_obs_2 \
--exp-name=img-obs-seq-200-stack \
--bert-model=bert-tiny --use-bert-encoder \
--batch-size=48 --add-norm-loss --epochs=1 \
--use-img-obs --traj-encoder=cnn --use-stack-img-obs

python -m lang_pref_learning.feature_learning.learn_features \
--initial-loss-check \
--data-dir=data/dataset_img_obs \
--exp-name=img-obs-seq-200-stack-small-net \
--bert-model=bert-tiny --use-bert-encoder \
--batch-size=64 --add-norm-loss --epochs=1 \
--use-img-obs --traj-encoder=cnn

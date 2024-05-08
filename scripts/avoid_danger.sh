python -m lang_pref_learning.feature_learning.learn_features \
--initial-loss-check \
--env=widowx \
--data-dir=data/data_avoid_danger \
--exp-name=widowx-img-obs \
--lang-model=t5-small --use-bert-encoder \
--batch-size=128 --add-norm-loss --epochs=1 \
--traj-reg-coeff=2e-2 --traj-reg-margin=1.0 --lr=1e-3 \
--use-img-obs --traj-encoder=cnn --seq-len=40

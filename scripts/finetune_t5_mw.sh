python -m lang_pref_learning.feature_learning.learn_features --initial-loss-check \
--env=franka-kitchen --data-dir=data/kitchen_2features \
--exp-name=mw-t5-small --lr=2e-3 --finetune-lr=2e-4 \
--lang-model-name=t5-small \
--batch-size=256 --add-norm-loss --epochs=1 \
--traj-encoder=mlp --seq-len=500 \
--traj-reg-coeff=1e-2 --traj-reg-margin=1.0

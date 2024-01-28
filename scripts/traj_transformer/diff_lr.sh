# Script to learn features from the data. Co-train BERT and the trajectory encoder.
for model in bert-tiny
do
  python -m feature_learning.learn_features --initial-loss-check --data-dir=data/dataset \
  --exp-name=traj-trans-diff-lr --bert-model=$model --use-bert-encoder --batch-size=16 --epochs=2 \
  --traj-encoder=transformer --set-different-lr
done
# Script to learn features from the data. Co-train BERT and the trajectory encoder.
for model in bert-tiny
do
  python -m feature_learning.learn_features --initial-loss-check --data-dir=data/dataset \
  --preprocessed-nlcomps --exp-name=traj-trans-lang-linear-$model --bert-model=$model --batch-size=32 --epochs=2 \
  --traj-encoder=transformer --lr=1e-3
done
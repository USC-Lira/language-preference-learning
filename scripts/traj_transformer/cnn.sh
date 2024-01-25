# Script to learn features from the data. Co-train BERT and the trajectory encoder.
for model in bert-tiny
do
  python -m feature_learning.learn_features --initial-loss-check --data-dir=data/dataset \
  --exp-name=traj-trans-lang-linear-$model --bert-model=$model --use-bert-encoder --batch-size=1024 --epochs=2 \
  --traj-encoder=transformer --use-cnn-in-transformer
done
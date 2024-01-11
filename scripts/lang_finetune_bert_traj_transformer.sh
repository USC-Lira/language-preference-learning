# Script to learn features from the data. Co-train BERT and the trajectory encoder.
for model in bert-tiny
do
  python -m feature_learning.learn_features --initial-loss-check --data-dir=data/dataset \
  --exp-name=traj-trans-lang-finetune-$model-newdata --bert-model=$model --use-bert-encoder --batch-size=512 --epochs=2 \
  --use-traj-transformer
done
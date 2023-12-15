# Script to learn features from the data. Co-train BERT and the trajectory encoder.
for model in bert-tiny bert-mini bert-base
do
  python -m feature_learning.learn_features --initial-loss-check --data-dir=data/dataset \
  --exp-name=finetune-$model-newdata --bert-model=$model --use-bert-encoder --batch-size=1024
done
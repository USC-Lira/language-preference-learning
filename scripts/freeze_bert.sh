# Script to learn features from the data
for model in bert-tiny bert-mini bert-base
do
  python -m feature_learning.learn_features --initial-loss-check \
  --data-dir=data/dataset --id-mapped --exp-name=freeze_$model_newdata --bert-model=$model &
done

wait
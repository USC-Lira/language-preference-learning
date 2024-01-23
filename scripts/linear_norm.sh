for model in bert-tiny bert-mini bert-base
do
  python -m feature_learning.learn_features --initial-loss-check --data-dir=data/dataset \
  --preprocessed-nlcomps --exp-name=linear-$model --bert-model=$model --batch-size=1024 --enable-norm-loss
done
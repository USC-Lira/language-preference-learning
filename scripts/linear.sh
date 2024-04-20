for model in bert-tiny
do
  python -m feature_learning.learn_features --initial-loss-check --data-dir=dataset_test \
  --preprocessed-nlcomps --exp-name=linear-$model --lang-model=$model --batch-size=1024
done
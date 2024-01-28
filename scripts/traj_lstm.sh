for model in bert-tiny
do
  python -m feature_learning.learn_features --initial-loss-check --data-dir=data/dataset \
  --preprocessed-nlcomps --exp-name=traj-lstm-lang-$model --bert-model=$model --batch-size=256 --traj-encoder=lstm
done
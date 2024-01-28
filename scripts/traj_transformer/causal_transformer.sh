for model in bert-tiny
do
  python -m feature_learning.learn_features --initial-loss-check --data-dir=data/dataset \
  --preprocessed-nlcomps --exp-name=traj-trans-lang-linear-$model --bert-model=$model --batch-size=256 --epochs=2 \
  --traj-encoder=transformer --use-cnn-in-transformer --use-casual-attention
done
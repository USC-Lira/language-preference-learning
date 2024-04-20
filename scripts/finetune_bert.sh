# Script to learn features from the data. Co-train BERT and the trajectory encoder.
for model in bert-tiny
do
  python -m lang_pref_learning.feature_learning.learn_features --initial-loss-check --data-dir=data/data_new_2 \
  --exp-name=finetune-$model-newdata --lang-model=$model --use-bert-encoder --batch-size=128 --epoch=1
done
# Script to learn features from the data. Co-train BERT and the trajectory encoder.
for model in bert-tiny bert-mini bert-base
do
  python -m lang_pref_learning.feature_learning.learn_features --initial-loss-check --data-dir=/scr/zyang966/data_new_2 \
  --exp-name=finetune-$model-newdata --bert-model=$model --use-bert-encoder --batch-size=1024 --epoch=1
done
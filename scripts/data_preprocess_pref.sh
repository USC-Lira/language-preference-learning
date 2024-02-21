# Script to learn features from the data, should be run for both train and val data
# Change data-dir to your data directory
# Example: bash scripts/data_preprocess.sh train
for model in bert-tiny
do
  python -m feature_learning.bert_preprocessing --id-mapping --data-dir=data/dataset-pref-learning/train --bert-model=$model
  python -m feature_learning.bert_preprocessing --id-mapping --data-dir=data/dataset-pref-learning/val --bert-model=$model
done

wait
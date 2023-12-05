# Script to learn features from the data, should be run for both train and val data
# Change data-dir to your data directory
# Example: bash scripts/data_preprocess.sh train
python -m feature_learning.bert_preprocessing --id-mapping \
--data-dir=data/$1
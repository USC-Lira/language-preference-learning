# Script to learn features from the data, should be run for both train and val data
# Change data-dir to your data directory
# Example: bash scripts/data_preprocess.sh train
for model in bert-tiny
do
  python -m lang_pref_learning.feature_learning.bert_preprocessing --id-mapping --data-dir=data/data_img_obs_res_224/train --bert-model=$model
  python -m lang_pref_learning.feature_learning.bert_preprocessing --id-mapping --data-dir=data/data_img_obs_res_224/val --bert-model=$model
  python -m lang_pref_learning.feature_learning.bert_preprocessing --id-mapping --data-dir=data/data_img_obs_res_224/test --bert-model=$model
done

wait
export OMP_NUM_THREADS=1

for model in bert-tiny
do
  python -m lang_pref_learning.feature_learning.bert_preprocessing --id-mapping --data-dir=data/metaworld/train --lang-model=$model
  python -m lang_pref_learning.feature_learning.bert_preprocessing --id-mapping --data-dir=data/metaworld/val --lang-model=$model
  python -m lang_pref_learning.feature_learning.bert_preprocessing --id-mapping --data-dir=data/metaworld/test --lang-model=$model
done

wait
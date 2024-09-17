export OMP_NUM_THREADS=1

for model in bert-tiny
do
  python -m lang_pref_learning.feature_learning.bert_preprocessing --id-mapping --data-dir=data/kitchen_2features/train --lang-model=$model
  python -m lang_pref_learning.feature_learning.bert_preprocessing --id-mapping --data-dir=data/kitchen_2features/val --lang-model=$model
  python -m lang_pref_learning.feature_learning.bert_preprocessing --id-mapping --data-dir=data/kitchen_2features/test --lang-model=$model
done

wait
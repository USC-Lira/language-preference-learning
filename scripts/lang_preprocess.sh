for model in bert-tiny
do
  python -m lang_pref_learning.feature_learning.bert_preprocessing --id-mapping --data-dir=data/data_avoid_danger/train --lang-model=$model
  python -m lang_pref_learning.feature_learning.bert_preprocessing --id-mapping --data-dir=data/data_avoid_danger/val --lang-model=$model
  python -m lang_pref_learning.feature_learning.bert_preprocessing --id-mapping --data-dir=data/data_avoid_danger/test --lang-model=$model
done

wait
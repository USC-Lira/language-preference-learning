for model in bert-tiny bert-mini bert-base
do
	python -m feature_learning.bert_preprocessing --id-mapping --data-dir=/scr/zyang966/data_new/train --bert-model=$model
	python -m feature_learning.bert_preprocessing --id-mapping --data-dir=/scr/zyang966/data_new/val --bert-model=$model
done

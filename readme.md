# Language Preference Learning

## Dependency
- python 3.8
- transformers
- torch

## Installation
```
# create conda environment
conda create -n lang python=3.8
conda activate lang

# install dependencies
pip install -r requirements.txt
```


## Generate BERT Embedding
```
python -m feature_learning.bert_preprocessing --id-mapping --data-dir=global_path_to_data_dir --bert-model=bert-base
```

## Feature Learning
```
python -m feature_learning.learn_features --data-dir=global_path_to_data_dir --id-mapped --preprocessed-nlcomps --bert-model=bert-base
```
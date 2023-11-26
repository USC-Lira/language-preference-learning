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
python -m feature_learning.bert_preprocessing --id-mapping --data-dir=global_path_to_data_dir
```

## Feature Learning
```
python -m feature_learning.feature_learning --data-dir=global_path_to_data_dir --id-mapped --preprocessed-nlcomps
```
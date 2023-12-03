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


## Generate Mapping from Trajectory Comparisons to Language Descriptions
```
python -m feature_learning.bert_preprocessing --id-mapping --data-dir=global_path_to_data_dir
```
Note that we use BERT as the language encoder, so there is no need to get language embeddings
in this step.

## Feature Learning
```
python -m feature_learning.learn_features --data-dir=global_path_to_data_dir --id-mapped
```
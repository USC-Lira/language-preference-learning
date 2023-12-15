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


## Data Preprocessing
```
python -m feature_learning.bert_preprocessing --id-mapping \
--data-dir=path_to_data_dir/train --bert-model=bert-base
```
Generate Mapping from Trajectory Comparisons to Language Descriptions. 
This should be run for both `train` and `val` folders. 

It will generate the index mapping from trajectory comparisons to language descriptions, 
and BERT embeddings for the language descriptions. 

## Feature Learning
### BERT + Linear layer
Only train the linear layer on top of BERT.
```
python -m feature_learning.learn_features --initial-loss-check \
--data-dir=path_to_data_dir --preprocessed-nlcomps --batch-size=1024 \
--exp-name=linear-bert-mini --bert-model=bert-mini 
```

### BERT only
When using BERT only, we adopt a two-stage training procedure. First, we freeze the BERT model and train the trajectory encoder. 
Then, we train the BERT model and the trajectory encoder jointly.
```
python -m feature_learning.learn_features --initial-loss-check \
--data-dir=data/dataset --batch-size=1024 \
--use-bert-encoder  --exp-name=finetune-bert-mini --bert-model=bert-mini 
```

Feel free to use the scripts in `scripts` folder to run the experiments!


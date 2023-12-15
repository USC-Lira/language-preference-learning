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
python -m feature_learning.bert_preprocessing --id-mapping --data-dir=global_path_to_data_dir --bert-model=bert-base
```
Generate Mapping from Trajectory Comparisons to Language Descriptions. 
This should be run for `train` and `test` folders. 

Note that we use BERT as the language encoder, so there is no need to get language embeddings
in this step.

## Feature Learning
First stage: freeze the language encoder and train the trajectory encoder
```
<<<<<<< HEAD
python -m feature_learning.learn_features --data-dir=global_path_to_data_dir --id-mapped --preprocessed-nlcomps --bert-model=bert-base
```
=======
python -m feature_learning.learn_features \
--data-dir=global_path_to_data_dir --id-mapped \
--exp-name=experiment_name
```

Second stage: train the language encoder and the trajectory encoder jointly
```
python -m feature_learning.learn_features \
--data-dir=path_to_data_dir --id-mapped \
--model-save-dir=path_to_model_save_dir \
--finetune-bert --lr=1e-5
```

Feel free to use the scripts in `scripts` folder to run the experiments!
>>>>>>> bert_mini

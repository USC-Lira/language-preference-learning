# Script to co-finetune BERT and the trajectory encoder
# Change data-dir to your data directory, and model-save-dir to the directory where you want to save the model
python -m feature_learning.learn_features --initial-loss-check \
--data-dir=data/ --id-mapped \
--model-save-dir=exp/freeze_bert_mini_20231130_191326 \
--finetune-bert --lr=0.00002
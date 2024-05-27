# export CUDA_VISIBLE_DEVICES=5
# Script to learn features from the data. Co-train BERT and the trajectory encoder.
for model in bert-tiny
do
  python -m lang_pref_learning.feature_learning.learn_features --initial-loss-check --data-dir=/scr/zyang966/language-preference-learning/data/data_img_obs_res_224_more \
  --exp-name=finetune-$model-newdata --lang-model=$model --use-bert-encoder --batch-size=128 --epoch=1 --lr=5e-3
done
python -m feature_learning.learn_features --initial-loss-check --data-dir=dataset_img_obs \
  --exp-name=finetune-bert-tiny-img-obs --bert-model=bert-tiny --use-bert-encoder --batch-size=256 --use-img-obs

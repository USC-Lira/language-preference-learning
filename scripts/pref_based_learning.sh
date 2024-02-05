#!/bin/bash
python -m pref_learning.pref_based_learning \
  --data-dir=/home/mdjun/language-preference-learning/data/dataset \
  --model-dir=/home/mdjun/language-preference-learning/feature_learning \
  --num-batches=2 \
  --preprocessed-nlcomps \
  --bert-model=bert-tiny \
  --use-bert-encoder \
  --use-cnn-in-transformer \

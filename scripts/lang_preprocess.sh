export OMP_NUM_THREADS=1

for model in t5-small
do
  python -m lang_pref_learning.feature_learning.lang_preprocessing --data-dir=data/kitchen_2features/train --lang-model=$model
  python -m lang_pref_learning.feature_learning.lang_preprocessing --data-dir=data/kitchen_2features/val --lang-model=$model
  python -m lang_pref_learning.feature_learning.lang_preprocessing --data-dir=data/kitchen_2features/test --lang-model=$model
done

wait
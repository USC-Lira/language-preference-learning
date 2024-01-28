# Param search for the number of heads and layers in the trajectory transformer
# export CUDA_VISIBLE_DEVICES=0

# Define the values for the number of heads and layers
lrs=(0.0001 0.001 0.01)

# Loop over each combination of num_heads and num_layers
for lr in "${lrs[@]}"; do
      echo "Running experiment with lr=$lr"

      # Run the experiment with the specified parameters
      python -m feature_learning.learn_features --initial-loss-check --data-dir=data/dataset \
      --preprocessed-nlcomps --bert-model=bert-tiny --batch-size=64 \
      --use-traj-transformer --exp-name=traj-trans-lr-$lr --lr=$lr

done
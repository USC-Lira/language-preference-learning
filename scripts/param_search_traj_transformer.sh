# Param search for the number of heads and layers in the trajectory transformer
# export CUDA_VISIBLE_DEVICES=0

# Define the values for the number of heads and layers
num_heads_values=(4 8)
num_layers_values=(2 3 4 6)

# Loop over each combination of num_heads and num_layers
for num_heads in "${num_heads_values[@]}"; do
    for num_layers in "${num_layers_values[@]}"; do
        echo "Running experiment with num_heads=$num_heads and num_layers=$num_layers"

        # Run the experiment with the specified parameters
        python -m feature_learning.learn_features --initial-loss-check --data-dir=data/dataset \
            --preprocessed-nlcomps --bert-model=bert-tiny --batch-size=256 \
            --n-heads=$num_heads --n-layers=$num_layers \
            --exp-name=traj-trans-h$num_heads-l$num_layers-lang-linear
    done
done
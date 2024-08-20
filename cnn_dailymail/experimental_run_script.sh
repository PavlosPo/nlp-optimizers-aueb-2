#!/bin/bash

# Function to run the Python script and clean up
run_and_clean() {
    local seed=$1
    echo "Running configuration with seed $seed..."
    PJRT_DEVICE=TPU python tune_xla_lightning.py --seed $seed --optim $optim --batch_size $batch_size
    
    echo "Waiting for 10 seconds..."
    sleep 10
    
    echo "Deleting checkpoints..."
    rm -rf ./checkpoints/*
    if [ $? -eq 0 ]; then
        echo "Checkpoints deleted successfully."
    else
        echo "Failed to delete checkpoints. Check permissions and paths."
    fi
    
    echo "Waiting another 5 seconds..."
    sleep 5
}

# Set common arguments
optim="adam"
batch_size=16

# Array of seeds
seeds=(1 10 100 1000)

# Loop through seeds and run the function
for seed in "${seeds[@]}"; do
    run_and_clean $seed
done

echo "All configurations completed."
#!/bin/bash

# Function to run the Python script and clean up
run_and_clean() {
    local seed=$1
    echo "Running configuration with seed $seed..."
    PJRT_DEVICE=TPU python main_xla_lightning_defaults.py --seed $seed --optim $optim --batch_size $batch_size
    
    echo "Waiting for 10 seconds..."
    sleep 10
    
    echo "Deleting checkpoints..."
    # This will match both 'checkpoints' and 'checkpoints_full_training'
    for dir in ./checkpoints*; do
        if [ -d "$dir" ]; then
            echo "Deleting contents of $dir"
            rm -rf "$dir"/*
            if [ $? -eq 0 ]; then
                echo "Contents of $dir deleted successfully."
            else
                echo "Failed to delete contents of $dir. Check permissions and paths."
            fi
        fi
    done
    
    echo "Waiting another 5 seconds..."
    sleep 5
}

# Set common arguments
optim="sgdm"
batch_size=16

# Array of seeds
seeds=(1 10 100 1000)

# Loop through seeds and run the function
for seed in "${seeds[@]}"; do
    run_and_clean $seed
done

echo "All configurations completed."
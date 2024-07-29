#!/bin/bash

# First set of arguments
seed1=1
optim="adabound"
batch_size=16

# Second set of arguments
seed2=10

# Third set of arguments
seed3=100

# Fourth set of arguments
seed4=1000

# Run the Python script with the first set of arguments
echo "Running first configuration..."
PJRT_DEVICE=TPU python tune_xla_lightning.py --seed $seed1 --optim $optim --batch_size $batch_size

# Wait for the Python script to finish
wait

# Add a delay to let the TPU clean itself
echo "Waiting for 10 seconds..."
sleep 10

# Run the Python script with the second set of arguments
echo "Running second configuration..."
PJRT_DEVICE=TPU python tune_xla_lightning.py --seed $seed2 --optim $optim --batch_size $batch_size

# Wait for the Python script to finish
wait

# Add a delay to let the TPU clean itself
echo "Waiting for 10 seconds..."
sleep 10

# Run the Python script with the Third set of arguments
echo "Running third configuration..."
PJRT_DEVICE=TPU python tune_xla_lightning.py --seed $seed3 --optim $optim --batch_size $batch_size

# Wait for the Python script to finish
wait

# Add a delay to let the TPU clean itself
echo "Waiting for 10 seconds..."
sleep 10

# Run the Python script with the Fourth set of arguments
echo "Running fourth configuration..."
PJRT_DEVICE=TPU python tune_xla_lightning.py --seed $seed4 --optim $optim --batch_size $batch_size

# Wait for the Python script to finish
wait

# Add a delay to let the TPU clean itself
echo "Waiting for 10 seconds..."
sleep 10
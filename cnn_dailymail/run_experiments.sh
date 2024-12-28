#!/bin/bash

echo 'Exploring models for optimizer: sgdm'

echo 'Running command: PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 10 --optim sgdm --batch_size 8 --learning_rate 8.475594220365433e-05 --training_mode lr_only_training_mode'
PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 10 --optim sgdm --batch_size 8 --learning_rate 8.475594220365433e-05 --training_mode lr_only_training_mode

echo 'Removing checkpoints in 10 seconds...'
sleep 10
rm -rf ./checkpoints**/
echo 'Killing python child processes in 5 seconds...'
sleep 5
sudo pkill -f python
sleep 5
if [ $? -ne 0 ]; then
    echo 'Error occurred. Exiting.'
    exit 1
fi

echo 'Running command: PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 1000 --optim sgdm --batch_size 8 --learning_rate 8.061368429365304e-05 --training_mode lr_only_training_mode'
PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 1000 --optim sgdm --batch_size 8 --learning_rate 8.061368429365304e-05 --training_mode lr_only_training_mode

echo 'Removing checkpoints in 10 seconds...'
sleep 10
rm -rf ./checkpoints**/
echo 'Killing python child processes in 5 seconds...'
sleep 5
sudo pkill -f python
sleep 5
if [ $? -ne 0 ]; then
    echo 'Error occurred. Exiting.'
    exit 1
fi

echo 'Running command: PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 100 --optim sgdm --batch_size 8 --learning_rate 7.178823679470453e-05 --training_mode lr_only_training_mode'
PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 100 --optim sgdm --batch_size 8 --learning_rate 7.178823679470453e-05 --training_mode lr_only_training_mode

echo 'Removing checkpoints in 10 seconds...'
sleep 10
rm -rf ./checkpoints**/
echo 'Killing python child processes in 5 seconds...'
sleep 5
sudo pkill -f python
sleep 5
if [ $? -ne 0 ]; then
    echo 'Error occurred. Exiting.'
    exit 1
fi

echo 'Running command: PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 1 --optim sgdm --batch_size 8 --learning_rate 0.0001305697975889018 --training_mode lr_only_training_mode'
PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 1 --optim sgdm --batch_size 8 --learning_rate 0.0001305697975889018 --training_mode lr_only_training_mode

echo 'Removing checkpoints in 10 seconds...'
sleep 10
rm -rf ./checkpoints**/
echo 'Killing python child processes in 5 seconds...'
sleep 5
sudo pkill -f python
sleep 5
if [ $? -ne 0 ]; then
    echo 'Error occurred. Exiting.'
    exit 1
fi

echo 'Finished exploring all configurations for optimizer: sgdm'

echo 'Exploring models for optimizer: adam'

echo 'Running command: PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 10 --optim adam --batch_size 8 --learning_rate 0.0009684041630190215 --training_mode lr_only_training_mode'
PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 10 --optim adam --batch_size 8 --learning_rate 0.0009684041630190215 --training_mode lr_only_training_mode

echo 'Removing checkpoints in 10 seconds...'
sleep 10
rm -rf ./checkpoints**/
echo 'Killing python child processes in 5 seconds...'
sleep 5
sudo pkill -f python
sleep 5
if [ $? -ne 0 ]; then
    echo 'Error occurred. Exiting.'
    exit 1
fi

echo 'Running command: PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 1000 --optim adam --batch_size 8 --learning_rate 0.0009214095846450699 --training_mode lr_only_training_mode'
PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 1000 --optim adam --batch_size 8 --learning_rate 0.0009214095846450699 --training_mode lr_only_training_mode

echo 'Removing checkpoints in 10 seconds...'
sleep 10
rm -rf ./checkpoints**/
echo 'Killing python child processes in 5 seconds...'
sleep 5
sudo pkill -f python
sleep 5
if [ $? -ne 0 ]; then
    echo 'Error occurred. Exiting.'
    exit 1
fi

echo 'Running command: PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 100 --optim adam --batch_size 8 --learning_rate 0.0007739764596944226 --training_mode lr_only_training_mode'
PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 100 --optim adam --batch_size 8 --learning_rate 0.0007739764596944226 --training_mode lr_only_training_mode

echo 'Removing checkpoints in 10 seconds...'
sleep 10
rm -rf ./checkpoints**/
echo 'Killing python child processes in 5 seconds...'
sleep 5
sudo pkill -f python
sleep 5
if [ $? -ne 0 ]; then
    echo 'Error occurred. Exiting.'
    exit 1
fi

echo 'Running command: PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 1 --optim adam --batch_size 8 --learning_rate 0.0008977580079737867 --training_mode lr_only_training_mode'
PJRT_DEVICE=TPU python main_xla_lightning_t5_v1_1.py --seed 1 --optim adam --batch_size 8 --learning_rate 0.0008977580079737867 --training_mode lr_only_training_mode

echo 'Removing checkpoints in 10 seconds...'
sleep 10
rm -rf ./checkpoints**/
echo 'Killing python child processes in 5 seconds...'
sleep 5
sudo pkill -f python
sleep 5
if [ $? -ne 0 ]; then
    echo 'Error occurred. Exiting.'
    exit 1
fi

echo 'Finished exploring all configurations for optimizer: adam'

echo 'Finished exploring all optimizers'

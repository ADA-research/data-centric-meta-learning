#!/bin/bash
#SBATCH --job-name=transfer
#SBATCH --mail-user=s2042096@vuw.leidenuniv.nl
#SBATCH --mail-type=FAIL
#SBATCH --output=log/%x_%A.out
#SBATCH --error=log/%x_%A.err
#SBATCH --mem=2G
#SBATCH --time=00:07:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

foundation=$1

cd /home/s2042096/data1/thesis/code

source ./slurms/setup.bash

source ./settings.bash

python3 ./scripts/transfer.py \
--foundation $foundation \
--target ${datasets[$SLURM_ARRAY_TASK_ID]} \
--exp_path $exp_path \
--num_target_classes $num_target_classes \
--epochs $transfer_epochs \
--batch_size $batch_size \
--seed $seed \
--folds $folds \
--lr $transfer_lr \
--momentum $transfer_momentum \
--test_size $test_size \
--architecture $architecture

echo "[$SHELL] #### Finished code."
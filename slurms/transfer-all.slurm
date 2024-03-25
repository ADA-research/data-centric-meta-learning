#!/bin/bash
#SBATCH --job-name=transfer
#SBATCH --array=0-899
#SBATCH --mail-user="s2042096@vuw.leidenuniv.nl"
#SBATCH --mail-type="FAIL"
#SBATCH --output=log/transfer/%x_%A-%a.out
#SBATCH --error=log/transfer/errors/%x_%A-%a.err
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

cd /home/s2042096/data1/thesis/code

source ./slurms/setup.bash

source ./settings.bash

echo "[$SHELL] #### started transfer"

python3 ./scripts/transfer.py \
--foundation ${datasets[$(($SLURM_ARRAY_TASK_ID / 30))]} \
--target ${datasets[$(($SLURM_ARRAY_TASK_ID % 30))]} \
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

echo "[$SHELL] #### Finished Python code."
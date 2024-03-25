#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mail-user=s2042096@vuw.leidenuniv.nl
#SBATCH --mail-type=FAIL
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

cd /home/s2042096/data1/thesis/code

source ./slurms/setup.bash

source ./settings.bash

ds=${datasets[$SLURM_ARRAY_TASK_ID]}

python3 ./scripts/train.py \
--dataset $ds \
--epochs $train_epochs \
--seed $seed \
--exp_path $exp_path \
--architecture $architecture \
--batch_size $batch_size \
--lr $train_lr \
--momentum $train_momentum \
--test_size $test_size

jid=$(sbatch -a "0-$num_datasets" -e "$errors_path/%x_%A-%a.err" -o "$output_path/%x_%A-%a.out" ./slurms/transfer.slurm $ds)

sbatch -d afterok:$jid -e "$errors_path/%x_%A-%a.err" -o "$output_path/%x_%A-%a.out" ./slurms/features.slurm $ds

echo "[$SHELL] #### Finished code."
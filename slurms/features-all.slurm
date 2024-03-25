#!/bin/bash
#SBATCH --job-name=features
#SBATCH --array=0-29
#SBATCH --output=log/%x_%A-%a.out
#SBATCH --error=log/%x_%A-%a.err
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

cd /home/s2042096/data1/thesis/code

source ./slurms/setup.bash

source ./settings.bash

python3 ./scripts/features.py \
--dataset ${datasets[$SLURM_ARRAY_TASK_ID]} \
--exp_path $exp_path \
--num_target_classes $num_target_classes \
--seed $seed \
--folds $folds \
--architecture $architecture

echo "[$SHELL] #### Finished code."
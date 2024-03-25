#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --output=log/%x_%A.out
#SBATCH --error=log/%x_%A.err
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1

cd /home/s2042096/data1/thesis/code

source ./slurms/setup.bash

source ./settings.bash

echo "[$SHELL] #### started prediction"
python3 ./scripts/predict.py \
--seed 1 \
--feature_set all \
--exp_path $exp_path \

echo "[$SHELL] #### Finished Python code."
#!/bin/bash
#SBATCH --job-name=features
#SBATCH --output=log/%x_%A.out
#SBATCH --error=log/%x_%A.err
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --partition=testing
#SBATCH --ntasks=1

cd /home/s2042096/data1/thesis/code

source ./slurms/setup.bash

source ./settings.bash

for ds in "${datasets[@]}"
do
    python3 ./scripts/features.py \
    --dataset $ds \
    --exp_path $exp_path \
    --num_target_classes $num_target_classes \
    --seed 2 \
    --folds $folds \
    --architecture $architecture
done
echo "[$SHELL] #### Finished code."
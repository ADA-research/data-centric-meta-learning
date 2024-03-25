#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mail-user=s2042096@vuw.leidenuniv.nl
#SBATCH --mail-type=FAIL
#SBATCH --output=log/%x_%A-%a.out
#SBATCH --error=log/%x_%A-%a.err
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

cd /home/s2042096/data1/thesis/code

# source ./slurms/setup.bash

# source ./settings.bash

# python3 ./scripts/test.py

tar cfj exp_data.tar.bz2 experiments/

echo "[$SHELL] #### Finished code."
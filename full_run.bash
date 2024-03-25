#!/bin/bash

cd /home/s2042096/data1/thesis/code

source ./settings.bash

mkdir $exp_path
mkdir $errors_path
mkdir $output_path

cp ./settings.bash $exp_path/configuration.bash

sbatch -a "0-$num_datasets" -e "$errors_path/%x_%A-%a.err" -o "$output_path/%x_%A-%a.out" ./slurms/train.slurm
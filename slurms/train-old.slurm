#!/bin/bash
#SBATCH --job-name=train
#SBATCH --array=0-29
#SBATCH --mail-user=2042096@vuw.leidenuniv.nl
#SBATCH --mail-type="FAIL"
#SBATCH --output=log/%x_%A-%a.out
#SBATCH --error=log/%x_%A-%a.err
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# navigate to root of system
cd /home/s2042096/data1/thesis/code

# setup environment
bash ./slurms/setup.bash

datasets=(
    "ACT_40_Mini"
    "ACT_410_Mini"
    "APL_Mini"
    "AWA_Mini"
    "BCT_Mini"
    "BRD_Mini"
    "BTS_Mini"
    "CRS_Mini"
    "DOG_Mini"
    "FLW_Mini"
    "FNG_Mini"
    "INS_2_Mini"
    "INS_Mini"
    "MD_5_BIS_Mini"
    "MD_6_Mini"
    "MD_MIX_Mini"
    "MED_LF_Mini"
    "PLK_Mini"
    "PLT_DOC_Mini"
    "PLT_NET_Mini"
    "PLT_VIL_Mini"
    "PNU_Mini"
    "PRT_Mini"
    "RESISC_Mini"
    "RSD_Mini"
    "RSICB_Mini"
    "SPT_Mini"
    "TEX_ALOT_Mini"
    "TEX_DTD_Mini"
    "TEX_Mini"
)


echo "[$SHELL] #### started training dataset ${datasets[$SLURM_ARRAY_TASK_ID]}"
python3 ./scripts/train.py --dataset ${datasets[$SLURM_ARRAY_TASK_ID]} --epochs 50 --seed 2

echo "[$SHELL] #### Finished Python code."
# making sure we start with a clean module environment
module purge

echo "## Starting GPU on $HOSTNAME"

echo "## Loading module"
module load slurm
module load CUDA/11.3.1
module load GCC/9.3.0

echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

# Create an environment variable set to where i want my environment to be placed
export ENV=/home/s2042096/data1/.conda/envs/thesis
export CWD=$(pwd)


# This setup is needed to find conda, you need this in all scripts where you want to use conda. Place this before activation.
__conda_setup="$('/cm/shared/easybuild/GenuineIntel/software/Miniconda3/4.9.2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cm/shared/easybuild/GenuineIntel/software/Miniconda3/4.9.2/etc/profile.d/conda.sh" ]; then
        . "/cm/shared/easybuild/GenuineIntel/software/Miniconda3/4.9.2/etc/profile.d/conda.sh"
    else
        export PATH="/cm/shared/easybuild/GenuineIntel/software/Miniconda3/4.9.2/bin:$PATH"
    fi
fi
unset __conda_setup

# Reset library path for conda
LD_LIBRARY_PATH=/data1/s2042096/.conda/envs/thesis/lib/

# Activating the environment
conda activate $ENV
echo "[$SHELL] ## conda env activated"

datasets=(
    "BRD_Mini"
    "DOG_Mini"
    "AWA_Mini"
    "PLK_Mini"
    "INS_2_Mini"
    "INS_Mini"
    "FLW_Mini"
    "PLT_NET_Mini"
    "FNG_Mini"
    "PLT_VIL_Mini"
    "MED_LF_Mini"
    "PLT_DOC_Mini"
    "BCT_Mini"
    "PNU_Mini"
    "PRT_Mini"
    "RESISC_Mini"
    "RSICB_Mini"
    "RSD_Mini"
    "CRS_Mini"
    "APL_Mini"
    "BTS_Mini"
    "TEX_Mini"
    "TEX_DTD_Mini"
    "TEX_ALOT_Mini"
    "SPT_Mini"
    "ACT_40_Mini"
    "ACT_410_Mini"
    "MD_MIX_Mini"
    "MD_5_BIS_Mini"
    "MD_6_Mini"
)
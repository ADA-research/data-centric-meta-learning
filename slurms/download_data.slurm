#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --partition=testing
#SBATCH --ntasks=1

cd /home/s2042096/data1/thesis/code

# Here I create an environment variable set to where i want my environment to be placed
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

LD_LIBRARY_PATH=/data1/s2042096/.conda/envs/thesis/lib/


# Activating the environment
conda activate $ENV
echo "[$SHELL] ## ***** conda env activated *****"

bash ./data/download_data.sh

echo "[$SHELL] #### Finished Python code."
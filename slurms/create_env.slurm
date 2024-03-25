#!/bin/bash
#SBATCH --job-name=creating_env_1
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --partition=testing
#SBATCH --ntasks=1

#Note: 
#   - The output is put in a log file in the directory of the slurm file. ./log must exist or first be created.
#   - Due to size limits I suggest placing the directory on the data1 disk, this way you won't get into memory troubles.

# Here I create an environment variable set to where i want my environment to be placed
ENV=/home/s2042096/data1/.conda/envs/thesis


echo "[$SHELL] #### Starting environment creation"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Printing some info for debugging if needed.
conda info

# Use this line if you want to remove the environment and build it from scratch.
# I suggest building environments from scratch if you want to install new libraries instead of installing with the command line.
conda env remove -p $ENV
echo "[$SHELL] ## ***** removed *****"


# Creating the environment
conda create --prefix $ENV
echo "[$SHELL] ## ***** created *****"


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


# Activating the environment
conda activate $ENV
echo "[$SHELL] ## ***** conda env activated *****"

# Again conda info for debugging if needed.
conda info

LD_LIBRARY_PATH=/data1/s2042096/.conda/envs/thesis/lib/

# This way you can directly install libraries, but I would recommand installing with an .yml file.
# conda install -p $ENV -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# Setting a library path, not 100% sure if needed anymore but it wont do any harm.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
echo "[$SHELL] ## after setting"


# Updating with an .yml file. This will download all libraries in that file.
conda env update -p $ENV --file $CWD/environment.yml
echo "***** yml file loaded *****"

echo "[$SHELL] #### Finished Python test. Have a nice day"
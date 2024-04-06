All scripts and code in this repository was written to facilitate the experiments that form my MSc research project.
The main purpose was to generate experimental results and data for further analysis. Unfortunately, maintainability and readability were beyond this scope. Apologies.

Nontheless, here is a minimal description of the important elements in this repository, for if they are ever to be reused in further research.

meta-data.csv is the final dataset that contains the results of my experiments and that I used for my analysis.

the /slurms directory contains all the slurm scripts that can be used in an environment that uses a slurm workload manager. The main idea behind the design is to allow different atomic parts of the experiments to run concurrently on a cluster.

the /scripts folder contains all the python scripts that make up the different procedures to train, fine-tune and create meta-models. Especially the scripts that create meta-models contain elements that are 'quick and dirty', as many last-minute adjustments were made to these procedures.

the /data folder contains scripts that can help you to download and decompress all the raw data of the meta-album dataset collection.

There is a settings.bash file where you can configure the settings of your experiments. Different bash scripts source this file. If you change the settings while a batch job is running, this can lead to unexpected results. There is definitively a better way to configure the experimental settings for bash jobs.
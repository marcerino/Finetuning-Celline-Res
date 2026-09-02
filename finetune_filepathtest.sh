#!/bin/bash
#SBATCH --job-name=fullgex_sizefilter_full_sweep
#SBATCH --output=/data/local/mgiller/atlas_tissue_representation/size_filterd/slurm-reports/flexynesis_%A_%a.out  # %A is job ID, %a is array index
#SBATCH --gpus=1
#SBATCH --array=0-7                   # Creates 12 jobs (indexes 0, 1, 2, 3, 4, 5, 6)

# In batch scripts, you often need to source your bashrc to make conda/mamba available
source ~/.bashrc
mamba activate flexenv

# Define the array of sample sizes
# Extract the specific sample size for this array task
#path for the test f textbased things
#/data/local/mgiller/atlas_tissue_representation/Data/more_than_25_samples/Finesamples-w-Error.txt 

# Run the command directly (no srun needed, the node and GPU are already allocated)
flexynesis \
    --pretrained_model /data/local/mgiller/atlas_tissue_representation/Models/Atlas-uberon_tissue-SupervisedVAE/fullfeatures.final_model.pth \
    --artifacts /data/local/mgiller/atlas_tissue_representation/Models/Atlas-uberon_tissue-SupervisedVAE/fullfeatures.artifacts.joblib \
    --data_path_test /data/local/mgiller/atlas_tissue_representation/Data/more_than_25_samples \
    --outdir /data/local/mgiller/atlas_tissue_representation/Data/more_than_25_samples/test \
    --target_variable uberon_tissue \
    --finetuning_samples 5  \
    --prefix test1
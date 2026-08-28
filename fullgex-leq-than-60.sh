#!/bin/bash
#SBATCH --job-name=fullgex_more_than_50samples_flexynesis_sweep
#SBATCH --output=/data/local/mgiller/atlas_tissue_representation/more_than_50samples/flexynesis_%A_%a.out  # %A is job ID, %a is array index
#SBATCH --gpus=1
#SBATCH --array=0-4                    # Creates 5 jobs (indexes 0, 1, 2, 3, 4)

# In batch scripts, you often need to source your bashrc to make conda/mamba available
source ~/.bashrc
mamba activate flexenv

# Define the array of sample sizes
SAMPLE_SIZES=(0 50 100 140 200)

# Extract the specific sample size for this array task
SAMPLES=${SAMPLE_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Running array task $SLURM_ARRAY_TASK_ID with $SAMPLES samples on $(hostname)"

# Run the command directly (no srun needed, the node and GPU are already allocated)
flexynesis \
    --pretrained_model /data/local/mgiller/atlas_tissue_representation/Models/Atlas-uberon_tissue-SupervisedVAE/fullfeatures.final_model.pth \
    --artifacts /data/local/mgiller/atlas_tissue_representation/Models/Atlas-uberon_tissue-SupervisedVAE/fullfeatures.artifacts.joblib \
    --data_path_test /data/local/mgiller/atlas_tissue_representation/Data/onco-cellinedata \
    --outdir /data/local/mgiller/atlas_tissue_representation/leq_than_60samples \
    --target_variable uberon_tissue \
    --finetuning_samples $SAMPLES \
    --prefix leq_than_60_samplesize$SAMPLES

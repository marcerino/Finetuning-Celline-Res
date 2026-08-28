#!/bin/bash
#SBATCH --job-name=flexynesis_sweep
#SBATCH --output=./fine-pred-claudemapping/finetunerun/flexynesis_%A_%a.out  # %A is job ID, %a is array index
#SBATCH --gpus=1
#SBATCH --array=0-6                    # Creates 7 jobs (indexes 0, 1, 2, 3, 4, 5, 6)

# In batch scripts, you often need to source your bashrc to make conda/mamba available
source ~/.bashrc
mamba activate flexenv

# Define the array of sample sizes
SAMPLE_SIZES=(0 50 100 150 200 300 400)

# Extract the specific sample size for this array task
SAMPLES=${SAMPLE_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Running array task $SLURM_ARRAY_TASK_ID with $SAMPLES samples on $(hostname)"

# Run the command directly (no srun needed, the node and GPU are already allocated)
flexynesis \
    --pretrained_model /data/local/mgiller/atlas_tissue_representation/Models/Atlas-uberon_tissue-SupervisedVAE/job.final_model.pth \
    --artifacts /data/local/mgiller/atlas_tissue_representation/Models/Atlas-uberon_tissue-SupervisedVAE/job.artifacts.joblib \
    --data_path_test /data/local/mgiller/atlas_tissue_representation/Data/celline-data \
    --outdir ./fine-pred-claudemapping \
    --target_variable uberon_tissue \
    --finetuning_samples $SAMPLES \
    --prefix samplezisze_$SAMPLES

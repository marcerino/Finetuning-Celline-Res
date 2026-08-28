#!/bin/bash
#SBATCH --job-name=fullgex_top10features_flexynesis_sweep
#SBATCH --output=/data/local/mgiller/atlas_tissue_representation/Models/low_data_split/flexynesis_%A_%a.out  # %A is job ID, %a is array index
#SBATCH --gpus=1
#SBATCH --array=0-3                    # Creates 4 jobs (indexes 0, 1,2,3)

# In batch scripts, you often need to source your bashrc to make conda/mamba available
source ~/.bashrc
mamba activate flexenv

# Define the array of sample sizes
MODEL_TYPES=(DirectPred supervised_vae DirectPred supervised_vae)
SUBSAMPLE=(108 108 50 50)

# Extract the specific sample size for this array task
MODEL=${MODEL_TYPES[$SLURM_ARRAY_TASK_ID]}
SAMPLES=${SUBSAMPLE[$SLURM_ARRAY_TASK_ID]}

echo "Running array task $SLURM_ARRAY_TASK_ID with $SAMPLES samples on $(hostname)"

# Run the command directly (no srun needed, the node and GPU are already allocated)
flexynesis \
    --data_path /data/local/mgiller/atlas_tissue_representation/Data/low-data-split \
    --data_types gex \
    --target_variable uberon_tissue \
    --model_class $MODEL \
    --outdir /data/local/mgiller/atlas_tissue_representation/Models/low_data_split/$MODEL \
    --evaluate_baseline_performance\
    --features_top_percentile 100 \
    --subsample $SAMPLES \
    --prefix $MODEL$SAMPLES

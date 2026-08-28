#!/bin/bash
#SBATCH --job-name=fullgex_top10features_flexynesis_sweep
#SBATCH --output=/data/local/mgiller/atlas_tissue_representation/Models/like_finetune_split/flexynesis_%A_%a.out  # %A is job ID, %a is array index
#SBATCH --gpus=1
#SBATCH --array=0-1                # Creates 6 jobs (indexes 0, 1,2,3)

# In batch scripts, you often need to source your bashrc to make conda/mamba available
source ~/.bashrc
mamba activate flexenv

# Define the array of sample sizes
SUBSAMPLES_LIST=(50 75)  #100 125 150 175 200

# Extract the specific sample size for this array task
SAMPLES=${SUBSAMPLES_LIST[$SLURM_ARRAY_TASK_ID]}
echo "Running array task $SLURM_ARRAY_TASK_ID with  $SAMPLES samples based on $(hostname)"

# Run the command directly (no srun needed, the node and GPU are already allocated)
for MODEL in supervised_vae  RandomForest
do
    flexynesis \
        --data_path /data/local/mgiller/atlas_tissue_representation/Data/${SAMPLES}_FinetuneSamples_split \
        --data_types gex \
        --target_variable uberon_tissue \
        --model_class $MODEL \
        --outdir /data/local/mgiller/atlas_tissue_representation/Models/like_finetune_split/ \
        --features_top_percentile 100 \
        --prefix from_scratch_${MODEL}_${SAMPLES}
done

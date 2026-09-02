#!/bin/bash
#SBATCH --job-name=fullgex_sizefilter_full_sweep
#SBATCH --output=/data/local/mgiller/atlas_tissue_representation/size_filterd/Resampled/flexynesis_%A_%a.out  # %A is job ID, %a is array index
#SBATCH --gpus=1
#SBATCH --array=0                  # Creates 7 jobs (indexes 0, 1, 2, 3, 4, 5, 6)

# In batch scripts, you often need to source your bashrc to make conda/mamba available
source ~/.bashrc
mamba activate flexenv

# Define the array of sample sizes
SAMPLE_SIZES=(0)
# Extract the specific sample size for this array task
SAMPLES=${SAMPLE_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Running array task $SLURM_ARRAY_TASK_ID with $SAMPLES samples on $(hostname)"

# Source - https://stackoverflow.com/a/49111
# Posted by Rob Rolnick, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-31, License - CC BY-SA 4.0

for i in $(seq 0 9);
do
    echo "Running array task $SLURM_ARRAY_TASK_ID with $SAMPLES Subsamplerun $i samples on $(hostname)"
    
    flexynesis \
        --pretrained_model /data/local/mgiller/atlas_tissue_representation/Models/Atlas-uberon_tissue-SupervisedVAE/fullfeatures.final_model.pth \
        --artifacts /data/local/mgiller/atlas_tissue_representation/Models/Atlas-uberon_tissue-SupervisedVAE/fullfeatures.artifacts.joblib \
        --data_path_test /data/local/mgiller/atlas_tissue_representation/Data/more_than_25_samples \
        --outdir /data/local/mgiller/atlas_tissue_representation/size_filterd/Resampled/resample_${i} \
        --target_variable uberon_tissue \
        --finetuning_samples 0 \
        --prefix 0_FinetuneSamples_resample_run_${i}

done

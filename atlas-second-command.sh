#!/bin/bash
#SBATCH --gpus=1

#.. any other jobs options here

# your command here

srun --gpus=1 --pty flexynesis \
  --data_path /data/local/mgiller/atlas_tissue_representation/processed_scaled_411k_tissue_B_h5 \
  --hpo_iter 1 \
  --model_class supervised_vae \
  --outdir ./Atlas-uberon_tissue-SupervisedVAE \
  --data_types gex \
  --target_variables uberon_tissue

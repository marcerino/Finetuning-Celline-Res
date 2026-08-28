#!/bin/bash
#SBATCH --gpus=1
#SBATCH --job-name=flexynesis_fullgex
#SBATCH --output=/data/local/mgiller/atlas_tissue_representation/Models/Atlas-uberon_tissue-SupervisedVAE/fullgex.out  # %A is job ID, %a is array index




# In batch scripts, you often need to source your bashrc to make conda/mamba available
source ~/.bashrc
mamba activate flexenv


#.. any other jobs options here

# your command here

flexynesis \
  --data_path /data/local/mgiller/atlas_tissue_representation/Data/processed_scaled_411k_tissue_B_h5 \
  --hpo_iter 1 \
  --model_class supervised_vae \
  --outdir /data/local/mgiller/atlas_tissue_representation/Models/Atlas-uberon_tissue-SupervisedVAE \
  --data_types gex \
  --features_top_percentile 100 \
  --target_variables uberon_tissue \
  --prefix fullfeatures

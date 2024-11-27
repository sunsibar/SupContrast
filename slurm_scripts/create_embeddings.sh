#!/usr/bin/env bash
#SBATCH --partition=a100-galvani
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=output/embeddings/%A_%a.txt
#SBATCH --error=output/embeddings/%A_%a.txt
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --signal=INT@600
#SBATCH --array=0-3  # Adjusted for 4 combinations

# Load configuration
source slurm_scripts/config.sh

if [[ -v singularity_img_path ]]; then
  echo "Loading from config.sh successful"
  echo "Project root: $project_root_on_server"
  echo "Singularity image: $singularity_img_path"
else
  echo "Error loading variables from config.sh"
  exit 1
fi

CUR_BASE_PATH=$project_root_on_server
SINGULARITY_COMMANDS=$additional_singularity_commands
ADD_CMDS=$additional_slurmscript_commands
$ADD_CMDS

scontrol show job "$SLURM_JOB_ID"
echo $PATH

# Define model types, datasets, and other parameters
MODEL_TYPES=("SupCon" "SimCLR" "SupCon" "SimCLR")
DATASETS=("cifar10" "cifar10" "cifar100" "cifar100")
MODEL="resnet34"  # Fixed model type

# Get the current task index
TASK_INDEX=$SLURM_ARRAY_TASK_ID
MODEL_TYPE=${MODEL_TYPES[$TASK_INDEX]}
DATASET=${DATASETS[$TASK_INDEX]}
# HEAD="--head"
HEAD=" "

NUM_EMBEDDINGS_PER_CLASS=-1  # Use entire dataset

LR=0.5
TEMPERATURE=0.1
if [[ $MODEL_TYPE == "SimCLR" ]]; then 
    TEMPERATURE=0.5
fi  

# Set checkpoint paths based on model type and dataset
CKPT="save/SupCon/${DATASET}_models/${MODEL_TYPE}_${DATASET}_${MODEL}_lr_${LR}_decay_0.0001_bsz_2048_temp_${TEMPERATURE}_trial_0_cosine_warm/ckpt_epoch_500.pth" 


echo "SLURM_ARRAY_TASK_ID: $TASK_INDEX"
echo "MODEL_TYPE: $MODEL_TYPE, MODEL: $MODEL, DATASET: $DATASET, CKPT: $CKPT, HEAD: $HEAD"

srun singularity exec -p --nv \
    --pwd /src/SupContrast \
    $SINGULARITY_COMMANDS \
    --bind $CUR_BASE_PATH:/src/SupContrast \
    $singularity_img_path \
    /usr/bin/python3.10 -u /src/SupContrast/create_embeddings.py \
        --model_type $MODEL_TYPE \
        --model_architecture $MODEL \
        --dataset $DATASET \
        --num_embeddings_per_class $NUM_EMBEDDINGS_PER_CLASS \
        --ckpt $CKPT \
        --output_dir ./embeddings \
        $HEAD # Use the full model output

# Directly create t-SNE plots
srun singularity exec -p --nv \
    --pwd /src/SupContrast \
    $SINGULARITY_COMMANDS \
    --bind $CUR_BASE_PATH:/src/SupContrast \
    $singularity_img_path \
    /usr/bin/python3.10 -u /src/SupContrast/analyses/tSNE_of_embeddings.py \
        --model_type $MODEL_TYPE \
        --model_architecture $MODEL \
        --dataset $DATASET \
        --num_embeddings_per_class 200 \
        --embeddings_dir ./embeddings \
        --output_dir ./analyses/plots/tSNE \
        $HEAD # Use the full model output

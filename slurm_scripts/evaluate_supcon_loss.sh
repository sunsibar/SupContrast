#!/usr/bin/env bash
#SBATCH --partition=a100-galvani
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=output/analyses/loss_evaluation/%A_%a.txt
#SBATCH --error=output/analyses/loss_evaluation/%A_%a.txt
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --signal=INT@600
#SBATCH --array=0-0

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
# MODEL_TYPES=("TargetVector" "TargetVector" "TargetVector" "TargetVector")
MODEL_TYPES=("SupCon" "SupCon" "SupCon" "SupCon")
DATASETS=("cifar100" "cifar10" "cifar100" "cifar10")
MODELS=("resnet34" "resnet34" "resnet18" "resnet18") 
# MODELS=("resnet18" "resnet18")
EPOCH=7500

# Get the current task index
TASK_INDEX=$SLURM_ARRAY_TASK_ID
MODEL_TYPE=${MODEL_TYPES[$TASK_INDEX]}
DATASET=${DATASETS[$TASK_INDEX]}
MODEL=${MODELS[$TASK_INDEX]}
# LR=0.5
# LR=0.15
LR=0.0002
TEMPERATURE=0.1
if [[ $MODEL_TYPE == "SimCLR" ]]; then 
    TEMPERATURE=0.5
fi  

# Set checkpoint path
CKPT="save/SupCon/${DATASET}_models/${MODEL_TYPE}_${DATASET}_${MODEL}_lr_${LR}_decay_0.0001_bsz_2048_temp_${TEMPERATURE}_trial_0_cosine_warm/ckpt_epoch_${EPOCH}.pth"

echo "SLURM_ARRAY_TASK_ID: $TASK_INDEX"
echo "MODEL_TYPE: $MODEL_TYPE, MODEL: $MODEL, DATASET: $DATASET, EPOCH: $EPOCH, CKPT: $CKPT"

# Run evaluation
srun singularity exec -p --nv \
    --pwd /src/SupContrast \
    $SINGULARITY_COMMANDS \
    --bind $CUR_BASE_PATH:/src/SupContrast \
    $singularity_img_path \
    /usr/bin/python3.10 -u /src/SupContrast/analyses/evaluate_supcon_loss.py \
        --model_type $MODEL_TYPE \
        --model_architecture $MODEL \
        --dataset $DATASET \
        --batch_size 2048 \
        --num_workers 8 \
        --temperature $TEMPERATURE \
        --ckpt $CKPT \
        --data_folder ./datasets
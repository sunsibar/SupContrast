#!/usr/bin/env bash
#SBATCH --partition=a100-galvani
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=output/linear_eval/%A_%a.txt
#SBATCH --error=output/linear_eval/%A_%a.txt
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --signal=INT@600
#SBATCH --array=3-3

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

# MODEL_TYPES=("SupCon" "SupCon" "SupCon" "SupCon")
MODEL_TYPES=("SimCLR" "SimCLR" "SimCLR" "SimCLR")
MODEL_TYPE=${MODEL_TYPES[$SLURM_ARRAY_TASK_ID]}
MODELS=("resnet18" "resnet18" "resnet34" "resnet34")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
DATASETS=("cifar10" "cifar100" "cifar10" "cifar100")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
LR=5
TEMPERATURE=0.1
if [[ $MODEL_TYPE == "SimCLR" ]]; then
    LR=1
    TEMPERATURE=0.5
fi  
CKPT=("save/SupCon/${DATASET}_models/${MODEL_TYPE}_${DATASET}_${MODEL}_lr_0.5_decay_0.0001_bsz_2048_temp_${TEMPERATURE}_trial_0_cosine_warm/ckpt_epoch_500.pth" )


echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "MODEL_TYPE: $MODEL_TYPE, MODEL: $MODEL, DATASET: $DATASET, LR: $LR, CKPT: $CKPT"


srun singularity exec -p --nv \
    --pwd /src/SupContrast \
    $SINGULARITY_COMMANDS \
    --bind $CUR_BASE_PATH:/src/SupContrast \
    $singularity_img_path \
    /usr/bin/python3.10 -u /src/SupContrast/main_linear.py \
        --batch_size 2048 \
        --learning_rate $LR \
        --dataset $DATASET \
        --num_workers 8 \
        --ckpt $CKPT \
        --epochs 100 \
        --model $MODEL
 

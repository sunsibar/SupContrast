#!/usr/bin/env bash
#SBATCH --partition=a100-galvani
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=output/simclr/%A_%a.txt
#SBATCH --error=output/simclr/%A_%a.txt
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --signal=INT@600
#SBATCH --array=0-3

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

DATASETS=("cifar10" "cifar100" "cifar10" "cifar100")
MODELS=("resnet18" "resnet18" "resnet34" "resnet34")

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

srun singularity exec -p --nv \
    --pwd /src/SupContrast \
    $SINGULARITY_COMMANDS \
    --bind $CUR_BASE_PATH:/src/SupContrast \
    $singularity_img_path \
    /usr/bin/python3.10 -u /src/SupContrast/main_supcon.py \
        --batch_size 2048 \
        --learning_rate 0.5 \
        --temp 0.5 \
        --cosine \
        --dataset $DATASET \
        --num_workers 8 \
        --model $MODEL \
        --epochs 1500 \
        --method SimCLR 
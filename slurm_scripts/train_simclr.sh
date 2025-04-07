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
#SBATCH --array=0-6

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

# DATASETS=("cifar10" "cifar100" "cifar10" "cifar100" "cifar10" "cifar100"  "cifar10" "cifar100"  "cifar10" "cifar100"  "cifar10" "cifar100"  "cifar10" "cifar100"  "cifar10" "cifar100")
# DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
# MODELS=("resnet18" "resnet18" "resnet34" "resnet34" "resnet34" "resnet34")
MODEL="resnet34" 
DATASET="cifar100"
EMB_DIMS=(1024 1024 1024 1024 1024 1024)
# EMB_DIMS=(1024 128 1024 128 1024 128)
# EMB_DIMS=(128 128 64 64 32 32 16 16)
# EMB_DIMS=(128 128 128 128 64 64 32 32 16 16 8 8 4 4 3 3)
# LABEL_SMOOTHING_LIST=(0.0 0.0 0.2 0.2 0.4 0.4)
# LABEL_SMOOTHING=${LABEL_SMOOTHING_LIST[$SLURM_ARRAY_TASK_ID]}
LABEL_SMOOTHING=0.0
CLIP_POS_LIST=(0.0 0.0 0.0 0.025 0.05 0.1 0.5)
CLIP_NEG_LIST=(0.0 0.1 0.5 0.0   0.0  0.0 0.0)
CLIP_POS=${CLIP_POS_LIST[$SLURM_ARRAY_TASK_ID]}
CLIP_NEG=${CLIP_NEG_LIST[$SLURM_ARRAY_TASK_ID]}

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
EMB_DIM=${EMB_DIMS[$SLURM_ARRAY_TASK_ID]}
MODEL_TYPE="SimCLR"
# LR_RELOAD=0.5
TEMPERATURE=0.5
# LR=0
NORM="batchnorm"
TRIAL="emb-dim-${EMB_DIM}_with-linear-eval-ls-${LABEL_SMOOTHING}"
LR=0.5
TRAIN_ON_NEG_ONLY=""
# TRAIN_ON_NEG_ONLY=" --train_on_neg_only "
# EPOCH_LIST=(2150 2250 2250 1950 850 2050)
# EPOCH=${EPOCH_LIST[$SLURM_ARRAY_TASK_ID]}


# TRIAL="emb-dim-${EMB_DIM}_with-linear-eval-ls-${LABEL_SMOOTHING}_cosine_warm"
# CKPT="save/SupCon/${DATASET}_models/${MODEL_TYPE}_${DATASET}_${MODEL}_lr_${LR}_decay_0.0001_bsz_2048_temp_${TEMPERATURE}_ls_${LABEL_SMOOTHING}_trial_${TRIAL}/ckpt_epoch_${EPOCH}.pth" 



echo "DATASET: $DATASET"
echo "MODEL: $MODEL"
echo "LR: $LR"
echo "TEMPERATURE: $TEMPERATURE"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "NORM: $NORM"
echo "TRIAL: $TRIAL"
echo "TRAIN_ON_NEG_ONLY: $TRAIN_ON_NEG_ONLY"
echo "EMB_DIM: $EMB_DIM"

srun singularity exec -p --nv \
    --pwd /src/SupContrast \
    $SINGULARITY_COMMANDS \
    --bind $CUR_BASE_PATH:/src/SupContrast \
    $singularity_img_path \
    /usr/bin/python3.10 -u /src/SupContrast/main_supcon.py \
        --batch_size 2048 \
        --learning_rate $LR \
        --temp $TEMPERATURE \
        --cosine \
        --dataset $DATASET \
        --num_workers 8 \
        --model $MODEL \
        --epochs 3000 \
        --method SimCLR \
        --trial $TRIAL \
        --emb_dim $EMB_DIM \
        --norm $NORM \
        --proj_head linear \
        $TRAIN_ON_NEG_ONLY \
        --linear_eval \
        --weight_decay 0.0001 \
        --label_smoothing $LABEL_SMOOTHING \
        --clip_pos $CLIP_POS --clip_neg $CLIP_NEG 


        # \
        # --ckpt $CKPT \
        # --reload_from_epoch $EPOCH


        # --increase_weight_decay \


        # --weight_decay 0.001 \
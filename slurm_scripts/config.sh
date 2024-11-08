#!/usr/bin/env bash
# * * THIS FILE WAS AUTOMATICALLY GENERATED * * 
# Dont' modify. Modify config_files/config.py instead.
mounted_project_root="/mnt/lustre/work/bethge/bkr857/projects/SupContrast/"
local_project_root="/src/SupContrast/"
project_root_on_server="/mnt/lustre/work/bethge/bkr857/projects/SupContrast"
singularity_img_path="/mnt/lustre/work/bethge/bkr857/projects/group-orbit-cl/singularity_imgs/pytorch-2.1.0-cuda11.8-cudnn8_v3.sif"
local_slurm_log_path="/src/SupContrast/output_slurm/"
remote_slurm_log_path="/mnt/lustre/work/bethge/bkr857/projects/SupContrast/output/"
additional_singularity_commands="--bind /mnt/lustre/work/bethge/bkr857/projects/SupContrast:/src/SupContrast  --bind /mnt/lustre/work/bethge/bkr857/projects/disco:/src/disco  --bind /mnt/lustre/work/bethge/bkr857/projects/2024_hackathon_locally_conn:/src/2024_hackathon_locally_conn    --bind /mnt/lustre/work/bethge/bkr857/other/singularity_homedir:/home   -H /mnt/lustre/work/bethge/bkr857/other/singularity_homedir  --bind $SCRATCH/.cache:$HOME/.cache  --bind $SCRATCH/.vscode-server:$HOME/.vscode-server  --bind $SCRATCH/.conda:$HOME/.conda  --bind $SCRATCH/.config:$HOME/.config  --bind $SCRATCH/.ipython:$HOME/.ipython  --bind $SCRATCH/.jupyter:$HOME/.jupyter  --bind $SCRATCH/.java:$HOME/.java  --bind $SCRATCH/.local:$HOME/.local  --bind $SCRATCH/.nv:$HOME/.nv  --bind $SCRATCH/.vim:$HOME/.vim  --bind $SCRATCH/.pki:$HOME/.pki  --bind $SCRATCH:$SCRATCH"
additional_slurmscript_commands="mkdir  -p $SCRATCH/.vscode-server  $SCRATCH/.pip  $SCRATCH/.cache  $SCRATCH/.conda  $SCRATCH/.config  $SCRATCH/.ipython  $SCRATCH/.java  $SCRATCH/.jupyter  $SCRATCH/.local  $SCRATCH/.nv  $SCRATCH/.vim  $SCRATCH/.pki"

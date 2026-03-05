#!/bin/bash

#SBATCH --job-name=Terramind
#SBATCH --account=PGS0413
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:03:00
#SBATCH --mem=16G
#SBATCH --output=/users/PGS0218/julina/projects/geography/damage_mapping_terramind/V2/slurm_logs/%x_%j.out
#SBATCH --error=/users/PGS0218/julina/projects/geography/damage_mapping_terramind/V2/slurm_logs/%x_%j.err

# USAGE:: sbatch --export=JOB_TYPE=train terramind.slurm
JOB_TYPE=${JOB_TYPE:-train}

module purge
module load cuda/12.4.1
module load miniconda3/24.1.2-py310
source activate lora

echo "====== JOB INFO ======"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Type: ${JOB_TYPE}"
echo "Working dir : $(pwd)"
echo "Hostname : $(hostname)"
echo "Date : $(date)"
echo "Python : $(which python3)"

echo "===== GPU INFO ====="
nvidia-smi
python3 - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
EOF



# WANDB_NAME="${JOB_TYPE}--juls"
proj_dir="/users/PGS0218/julina/projects/geography/damage_mapping_terramind/V2/"

echo "Executing command for job type: $JOB_TYPE"
case $JOB_TYPE in
  train)
    # torchrun --nproc_per_node=1 ../damage_mapping/run_exp.py --config-name=train --wandb_name="$wandb_name"
    python3 ${proj_dir}/damage_mapping/trainer.py
    ;;
    
  evaluate)
    CUDA_VISIBLE_DEVICES=0 accelerate launch ${proj_dir}/damage_mapping/run_exp.py --config-name=evaluate --wandb_name="$WANDB_NAME"
    ;;

  *)
    echo "ERROR: Invalid job type '$JOB_TYPE'. Valid options: {train|evaluate}"
    ;;
esac

echo "====== Job '$JOB_TYPE' finished at $(date) ======"
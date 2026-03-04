#!/bin/bash -l

#SBATCH --job-name="Terramind"
#SBATCH --account=PGS0413
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --proj-dir="/users/PGS0218/julina/projects/geography/damage_mapping_terramind/V2/"


# USAGE:: sbatch terramind.slurm --job_type== <train|evaluate>


LOG_FILE_PATH="${proj-dir}"/data/output/slurm_logs"/${JOB_TYPE}_${SLURM_JOB_ID}_${date}.out"
exec > "$LOG_FILE_PATH" 2>&1

module purge
module load cuda/12.4.1
module load miniconda3/24.1.2-py310
source activate lora

cho "====== JOB INFO ======"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Type: ${JOB_TYPE}"
echo "Working dir : $(pwd)"
echo "Hostname : $(hostname)"
echo "Date : $(date)"
echo "Python : $(which python)"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "Git branch : $(git rev-parse --abbrev-ref HEAD)"   # <- Added branch info
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPUs detected: $(python -c 'import torch; print(torch.cuda.device_count())')"


WANDB_NAME="{JOB_TYPE}-${ITERS}-ep${EPOCHS}_${REG_ALPHA}"
# cd ${proj-dir}


echo "Executing command for job type: $JOB_TYPE"
case $JOB_TYPE in
  train)
    torchrun --nproc_per_node=1 ../damage_mapping/run_exp.py --config-name=train --wandb_name="$wandb_name"
    ;;
    
  evaluate)
    CUDA_VISIBLE_DEVICES=0 accelerate launch ../damage_mapping/run_exp.py --config-name=evaluate --wandb_name="$wandb_name"
    ;;

  *)
    echo "ERROR: Invalid job type '$JOB_TYPE'."
    echo "Valid options are: {train|evaluate}"
    ;;
esac

echo
echo "====== Job '$JOB_TYPE' finished successfully at '$(date)'. ======"

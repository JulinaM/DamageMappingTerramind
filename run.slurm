#!/bin/bash

#SBATCH --job-name="Terramind"
#SBATCH --account=pgs0385
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=64G

#USAGE::
# sbatch run.slurm --job_type=Train 

JOB_TYPE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --job_type=*|--job_type)
      JOB_TYPE="${1#*=}"; [ "$JOB_TYPE" = "$1" ] && { JOB_TYPE="$2"; shift; }
      ;;
    -*)
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      break
      ;;
  esac
  shift
done

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_TYPE}_${SLURM_JOB_ID}_${date}.out"
exec > "$LOG_FILE" 2>&1

# --- Load modules and activate environment ---
module purge
module load cuda/12.4.1
module load miniconda3/24.1.2-py310
source activate lora

cd /users/PGS0218/julina/projects/geography/damage_mapping_terramind/V2/

cho "====== JOB INFO ======"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Type: $JOB_TYPE"
echo "Working dir : $(pwd)"
echo "Hostname : $(hostname)"
echo "Date : $(date)"
echo "Python : $(which python)"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "Git branch : $(git rev-parse --abbrev-ref HEAD)"   # <- Added branch info
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPUs detected: $(python -c 'import torch; print(torch.cuda.device_count())')"


echo "Executing command for job type: $JOB_TYPE"
case $JOB_TYPE in
  train)
    CUDA_VISIBLE_DEVICES=0 accelerate launch Train.py 
        -m \
    ;;
    # torchrun --nproc_per_node=2 eval_gsm8k.py --model_name="$ckpt" --wandb_name="$wandb_name"


  test)
    CUDA_VISIBLE_DEVICES=0 accelerate launch Test.py \
    -m \
    ;;

  *)
    echo "ERROR: Invalid job type '$JOB_TYPE'."
    echo "Valid options are: {Train|Test}"
    ;;
esac

echo
echo "====== Job '$JOB_TYPE' finished successfully at '$(date)'. ======"

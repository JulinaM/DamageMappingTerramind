#!/bin/bash

#SBATCH --job-name=Terramind
#SBATCH --account=PGS0413
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -Eeuo pipefail

## USAGE::
# Minimal (default): sbatch --export=JOB_TYPE=train_large slurm/terramind.slurm
# Debug infra logs: sbatch --export=JOB_TYPE=train,SLURM_LOG_LEVEL=debug slurm/terramind.slurm

JOB_TYPE=${JOB_TYPE:-train_small} #train_small|train_large|train_debug
SLURM_LOG_LEVEL="${SLURM_LOG_LEVEL:-minimal}"  # minimal|debug
PROJ_DIR="/users/PGS0218/julina/projects/geography/damage_mapping_terramind/V2"
LOG_DIR="${PROJ_DIR}/data/slurm_logs"

case "${JOB_TYPE}" in
  train_old) LOG_STEM="terramind-train-old" ;;
  *) LOG_STEM="terramind-${JOB_TYPE//_/-}" ;;
esac

mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/${LOG_STEM}_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/${LOG_STEM}_${SLURM_JOB_ID}.err"

timestamp() { date +"%Y-%m-%d %H:%M:%S %Z"; }
log() { echo "[SLURM $(timestamp)] $*"; }

on_error() {
  local exit_code=$?
  >&2 echo "[SLURM $(timestamp)] ERROR: command failed (exit=${exit_code}) at line ${BASH_LINENO[0]}"
  >&2 echo "[SLURM $(timestamp)] ERROR: ${BASH_COMMAND}"
  >&2 echo "[SLURM $(timestamp)] Job '${JOB_TYPE}' failed"
  exit "${exit_code}"
}
trap on_error ERR

# Reproducible environment setup.
cd "${PROJ_DIR}"
module --force purge
module load cuda/12.4.1 miniconda3/24.1.2-py310
source activate lora


log ">> Job start | id=${SLURM_JOB_ID} type=${JOB_TYPE} host=$(hostname)"

if [[ "${SLURM_LOG_LEVEL}" == "debug" ]]; then
  echo
  log "pwd=$(pwd)"
  log "python=$(which python3)"
  nvidia-smi
fi

# Better traceback formatting for Hydra exceptions.
export HYDRA_FULL_ERROR=1

case $JOB_TYPE in
  train_large)
    CMD=(
      python3 -u -m damage_mapping
      "job_type=${JOB_TYPE}"
      "experiment_name=dev-run-large"
      "model.modalities=[S2L2A]"
      "~train_loader.modalities.S1GRD"
      "~validation_loader.modalities.S1GRD"
    )
    ;;
  train_small)
    CMD=(
      python3 -u -m damage_mapping 
      "--config-name=terramind_small"
      "experiment_name=dev-run-small"
    )
    ;;

  train_old)
    CMD=(
      python3 -u "${PROJ_DIR}/damage_mapping/trainer.py"
      "model.modalities=[S2L2A]"
      "~train_loader.modalities.S1GRD"
      "~validation_loader.modalities.S1GRD"
      "model.num_classes=4"
    )
    ;;

  *)
    >&2 echo "ERROR: Invalid job type '${JOB_TYPE}'. Valid options: {train_large|train_small|train_debug|eval}"
    exit 2
    ;;
esac

if [[ "${SLURM_LOG_LEVEL}" == "debug" ]]; then
  log "cmd=${CMD[*]}"
fi
"${CMD[@]}"

echo
echo "====== Job id=${SLURM_JOB_ID}  '$JOB_TYPE' finished successfully. ======"

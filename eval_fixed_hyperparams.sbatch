#!/bin/bash
#
#SBATCH --job-name=cleanrl-fixed-eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=aalaeddini_391_0001
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=duguezack@gmail.com
#
# #SBATCH --mem=64G

set -euo pipefail

# Print useful error context if the job fails.
trap 'echo ""; echo "========== JOB FAILED =========="; echo "Exit code: $?"; echo "Line: $LINENO"; echo "Working dir: $(pwd)"; echo "Date: $(date)"; echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"; echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-none}"; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"; echo "================================";' ERR

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT="cleanrl/ppo_atari_envpool.py"
EVAL_LAUNCHER="eval_fixed_hparams_envpool.py"

OPTIMIZERS="Adam,Muon"

LEARNING_RATE="0.00075"
AUX_LEARNING_RATE="0.00075"
ENT_COEF="0.01"
MOMENTUM="0.95"
UPDATE_EPOCHS="5"

TOTAL_TIMESTEPS="10000000"
NUM_ENVS="32"
NUM_STEPS="256"

NUM_SEEDS="10"
SEED_START="1"

GPUS="0,1,2,3,4,5,6,7"
CPUS_PER_RUN="16"

WANDB_PROJECT="cleanrl-atari-fixed-hparam-eval"
WANDB_TAG_PREFIX="fixed_lr00075_aux00075_ent001_mom095_10M"

LOGS_ROOT="eval_logs/${WANDB_TAG_PREFIX}"

mkdir -p logs "${LOGS_ROOT}"

DEBUG_LOG="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.debug.log"
exec > >(tee -a "${DEBUG_LOG}") 2>&1

echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "SCRIPT=${SCRIPT}"
echo "EVAL_LAUNCHER=${EVAL_LAUNCHER}"
echo "OPTIMIZERS=${OPTIMIZERS}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "AUX_LEARNING_RATE=${AUX_LEARNING_RATE}"
echo "ENT_COEF=${ENT_COEF}"
echo "MOMENTUM=${MOMENTUM}"
echo "TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_TAG_PREFIX=${WANDB_TAG_PREFIX}"
echo "LOGS_ROOT=${LOGS_ROOT}"
echo "DEBUG_LOG=${DEBUG_LOG}"

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

module purge

module load conda || true
source ~/.bashrc
conda activate RLMuon

cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f "${EVAL_LAUNCHER}" ]]; then
    echo "ERROR: EVAL_LAUNCHER does not exist:"
    echo "  ${EVAL_LAUNCHER}"
    exit 1
fi

if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: CleanRL script does not exist:"
    echo "  ${SCRIPT}"
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export TF_CPP_MIN_LOG_LEVEL=2

export OMP_NUM_THREADS="${CPUS_PER_RUN}"
export MKL_NUM_THREADS="${CPUS_PER_RUN}"
export OPENBLAS_NUM_THREADS="${CPUS_PER_RUN}"
export NUMEXPR_NUM_THREADS="${CPUS_PER_RUN}"

export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300

python - <<'PY'
import os
import torch
print("[info] torch.cuda.is_available() =", torch.cuda.is_available())
print("[info] torch.cuda.device_count() =", torch.cuda.device_count())
print("[info] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[info] OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
print("[info] MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
PY

# ─────────────────────────────────────────────────────────────────────────────
# Run fixed-hparam evaluation
# ─────────────────────────────────────────────────────────────────────────────

python -u "${EVAL_LAUNCHER}" \
  --script "${SCRIPT}" \
  --optimizers "${OPTIMIZERS}" \
  --gpus "${GPUS}" \
  --num-seeds "${NUM_SEEDS}" \
  --seed-start "${SEED_START}" \
  --game-set eval30 \
  --learning-rate "${LEARNING_RATE}" \
  --aux-learning-rate "${AUX_LEARNING_RATE}" \
  --ent-coef "${ENT_COEF}" \
  --momentum "${MOMENTUM}" \
  --update-epochs "${UPDATE_EPOCHS}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --num-envs "${NUM_ENVS}" \
  --num-steps "${NUM_STEPS}" \
  --wandb-project-name "${WANDB_PROJECT}" \
  --wandb-tag-prefix "${WANDB_TAG_PREFIX}" \
  --logs-root "${LOGS_ROOT}" \
  --cpus-per-run "${CPUS_PER_RUN}"

echo "Job finished on $(date)"
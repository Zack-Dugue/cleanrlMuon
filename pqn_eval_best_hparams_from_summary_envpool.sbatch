#!/bin/bash
#
#SBATCH --job-name=pqn-best-eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=aalaeddini_391_0001
#
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=zdugue@smu.edu
#
# #SBATCH --mem=64G

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT="cleanrl/pqn_atari_envpool_wandb.py"
EVAL_LAUNCHER="eval_best_pqn_hparams_envpool.py"

# Source of truth for best PQN hyperparameters.
# Use the best_hyperparams.txt/json produced by your PQN tuner.
BEST_SUMMARY_FILE="tuner_logs/atari10_pqn_envpool_pqn_muon_fixed/best_hyperparams.txt"

# Must match the optimizer string accepted by your PQN script.
OPTIMIZER="Muon"

NUM_SEEDS=10
SEED_START=1

# Match PQN tuning unless intentionally changing eval budget.
TOTAL_TIMESTEPS=20000000
NUM_ENVS=256
NUM_STEPS=32
NUM_MINIBATCHES=4
UPDATE_EPOCHS=4
MOMENTUM=0.95

GPUS="0,1,2,3,4,5,6,7"
CPUS_PER_RUN=16

WANDB_PROJECT="cleanrl-atari-best-hparam-eval"
WANDB_TAG="pqn_${OPTIMIZER}_best_eval"

LOGS_ROOT="eval_logs/${WANDB_TAG}"

mkdir -p logs "${LOGS_ROOT}"

echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "SCRIPT=${SCRIPT}"
echo "EVAL_LAUNCHER=${EVAL_LAUNCHER}"
echo "BEST_SUMMARY_FILE=${BEST_SUMMARY_FILE}"
echo "OPTIMIZER=${OPTIMIZER}"
echo "TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "NUM_STEPS=${NUM_STEPS}"
echo "NUM_MINIBATCHES=${NUM_MINIBATCHES}"
echo "UPDATE_EPOCHS=${UPDATE_EPOCHS}"
echo "MOMENTUM=${MOMENTUM}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_TAG=${WANDB_TAG}"
echo "LOGS_ROOT=${LOGS_ROOT}"

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

module purge
module load conda || true
source ~/.bashrc
conda activate RLMuon

cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f "${BEST_SUMMARY_FILE}" ]]; then
    echo "ERROR: BEST_SUMMARY_FILE does not exist:"
    echo "  ${BEST_SUMMARY_FILE}"
    exit 1
fi

if [[ ! -f "${EVAL_LAUNCHER}" ]]; then
    echo "ERROR: EVAL_LAUNCHER does not exist:"
    echo "  ${EVAL_LAUNCHER}"
    exit 1
fi

if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: CleanRL PQN script does not exist:"
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
# Run evaluation
# ─────────────────────────────────────────────────────────────────────────────

python -u "${EVAL_LAUNCHER}" \
  --script "${SCRIPT}" \
  --best-summary-file "${BEST_SUMMARY_FILE}" \
  --optimizer "${OPTIMIZER}" \
  --gpus "${GPUS}" \
  --num-seeds "${NUM_SEEDS}" \
  --seed-start "${SEED_START}" \
  --game-set eval30 \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --num-envs "${NUM_ENVS}" \
  --num-steps "${NUM_STEPS}" \
  --num-minibatches "${NUM_MINIBATCHES}" \
  --update-epochs "${UPDATE_EPOCHS}" \
  --momentum "${MOMENTUM}" \
  --wandb-project-name "${WANDB_PROJECT}" \
  --wandb-tag "${WANDB_TAG}" \
  --logs-root "${LOGS_ROOT}" \
  --cpus-per-run "${CPUS_PER_RUN}" \
  --fail-fast

echo "Job finished on $(date)"
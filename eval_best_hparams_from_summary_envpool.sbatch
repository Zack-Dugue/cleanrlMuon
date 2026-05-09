#!/bin/bash
#
#SBATCH --job-name=cleanrl-best-eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=aalaeddini_391_0001
# #SBATCH --mem=64G

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT="cleanrl/ppo_atari_envpool.py"
EVAL_LAUNCHER="best_run_eval.py"

# This is now the source of truth for the best hyperparameters.
# It should be the summary .txt written by write_optuna_summary_file().
BEST_SUMMARY_FILE="tuner_logs/atari10_envpool_RLMuon/optuna_summaries/atari10_envpool_RLMuon__RL_Muon_test.txt"

# Optimizer to pass through to CleanRL.
# This should match the optimizer used during the tuning run.
OPTIMIZER="RLMuon"

# Evaluation size.
NUM_SEEDS=10
SEED_START=1

# Keep these the same as tuning unless you intentionally want a different eval budget.
TOTAL_TIMESTEPS=5000000
NUM_ENVS=32
NUM_STEPS=256

# 8 GPUs per node.
GPUS="0,1,2,3,4,5,6,7"

# With 8 concurrent runs on 128 CPUs, 16 CPU threads per run is natural.
CPUS_PER_RUN=16

# Separate W&B project from tuning.
WANDB_PROJECT="cleanrl-atari-best-hparam-eval"

# Keep this descriptive and unique.
WANDB_TAG="${OPTIMIZER}_best_summary_eval"

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
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_TAG=${WANDB_TAG}"
echo "LOGS_ROOT=${LOGS_ROOT}"

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

module purge

# Adjust these two lines to match however you normally activate this env.
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
    echo "ERROR: CleanRL script does not exist:"
    echo "  ${SCRIPT}"
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export TF_CPP_MIN_LOG_LEVEL=2

# These are launcher-level defaults. The Python launcher also sets these
# per subprocess.
export OMP_NUM_THREADS="${CPUS_PER_RUN}"
export MKL_NUM_THREADS="${CPUS_PER_RUN}"
export OPENBLAS_NUM_THREADS="${CPUS_PER_RUN}"
export NUMEXPR_NUM_THREADS="${CPUS_PER_RUN}"

# Helps W&B under subprocess-heavy runs.
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
  --wandb-project-name "${WANDB_PROJECT}" \
  --wandb-tag "${WANDB_TAG}" \
  --logs-root "${LOGS_ROOT}" \
  --cpus-per-run "${CPUS_PER_RUN}"

echo "Job finished on $(date)"
#!/bin/bash
#
#SBATCH --job-name=cleanrl-pqn-tuner
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

# ── (0) User config ───────────────────────────────────────────────────────────
OPTIMIZER="Muon"
TRIALS=20
SEEDS=3
LOGS_ROOT="logs"
STORAGE_URL="sqlite:///cleanrl_hpopt.db"
WANDB_TAG="muon_input"

# PQN trainer entrypoint
SCRIPT="cleanrl/pqn_atari_envpool.py"

# Tuner wrapper
TUNER="pqn_tuner_example.py"

# Study name
STUDY_NAME="atari10_pqn_envpool_${OPTIMIZER}"

# ── (1) Basic setup ───────────────────────────────────────────────────────────
mkdir -p logs
echo "Job started on $(date) at host $(hostname)"
echo "SLURM_JOBID=${SLURM_JOB_ID:-"(none)"}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-"(none)"}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"(unset)"}"

module purge
module load conda
module load miniforge/24.11.2-1

source ~/.bashrc
conda activate RLMuon

export TF_CPP_MIN_LOG_LEVEL=2
export LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Keep the launcher light; worker thread counts are handled inside the tuner.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export BLIS_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"

# ── (2) Sanity / derive local GPU ids 0..N-1 ─────────────────────────────────
python - <<'PY'
import os, torch
print("[info] Host:", os.uname().nodename if hasattr(os, "uname") else "unknown")
print("[info] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[info] torch.cuda.device_count() =", torch.cuda.device_count())
print("[info] visible local GPU ids =", list(range(torch.cuda.device_count())))
PY

GPU_COUNT=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

if [ "${GPU_COUNT}" -le 0 ]; then
  echo "[fatal] No GPUs visible to PyTorch."
  exit 1
fi

GPUS=$(python - <<'PY'
import torch
print(",".join(str(i) for i in range(torch.cuda.device_count())))
PY
)

echo "[info] Using local GPU ids: ${GPUS}"

# ── (3) Launch PQN tuner ──────────────────────────────────────────────────────
echo "[info] Starting ${TUNER}"
echo "[info] SCRIPT=${SCRIPT}"
echo "[info] OPTIMIZER=${OPTIMIZER}"
echo "[info] STUDY_NAME=${STUDY_NAME}"
echo "[info] TRIALS=${TRIALS} SEEDS=${SEEDS}"

srun python -u "${TUNER}" \
  --script "${SCRIPT}" \
  --gpus "${GPUS}" \
  --optimizer "${OPTIMIZER}" \
  --trials "${TRIALS}" \
  --seeds "${SEEDS}" \
  --study-name "${STUDY_NAME}" \
  --storage "${STORAGE_URL}" \
  --logs-root "${LOGS_ROOT}" \
  --wandb-tag "${WANDB_TAG}"

echo "Job finished on $(date)"
#!/bin/bash
#
#SBATCH --job-name=cleanrl-tuner
#SBATCH --partition=batch              # change if your cluster uses a different queue
#SBATCH --nodes=1
#SBATCH --ntasks=1                     # one launcher; tuner will spawn per-GPU workers
#SBATCH --cpus-per-task=128            # total CPU threads available to the job
#SBATCH --gpus=8                       # request two GPUs
#SBATCH --time=24:00:00                # wall time
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=aalaeddini_391_0001
# # Optional memory line (uncomment/tune if your cluster enforces it)
# #SBATCH --mem=64G

set -eo pipefail

# ── (0) Config you can tweak right here ───────────────────────────────────────
OPTIMIZER="Muon"                              # ← choose: adamw | adam | muon | etc. (your script handles it)
GPUS="0,1,2,3,4,5,6,7"                        # local GPU indices (Slurm masks devices per node)
TRIALS=20
SEEDS=3
STUDY_NAME="atari10_envpool_${OPTIMIZER}"
WANDB_TAG="muon_input"                        # ← put your WandB tag here (or set to "" for none)
LOGS_ROOT="logs"
STORAGE_URL="sqlite:///cleanrl_hpopt.db"     # use MySQL/Postgres for multi-node

# Path to your EnvPool-based CleanRL script
SCRIPT="cleanrl/ppo_atari_envpool.py"

# ── (1) Basic runtime setup ───────────────────────────────────────────────────
mkdir -p logs
echo "Job started on $(date) at host $(hostname)"
echo "SLURM_JOBID=$SLURM_JOB_ID  SLURM_NODELIST=$SLURM_NODELIST"
echo "GPUs allocated (per Slurm): ${CUDA_VISIBLE_DEVICES:-"(assigned by Slurm)"}"


# ===== Activate your environment =====
# If your site uses modules, load them here (example):
module purge
module load conda
# Or just conda:
source ~/.bashrc
conda init
conda activate RLMuon   # TODO: set your conda env

# Sanity: what did we get?
echo "[info] Host: $(hostname)"
echo "[info] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"(unset)"}"
python - <<'PY'
import torch, os
print("[info] torch.cuda.device_count() =", torch.cuda.device_count())
print("[info] GPUs visible:", list(range(torch.cuda.device_count())))
print("[info] OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
PY


# Optional: quieten TF/CUDA logs; switch to INFO for less verbosity
export TF_CPP_MIN_LOG_LEVEL=2
export LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Actually create the env
module load miniforge/24.11.2-1
conda activate RLMuon

# ── (3) Move to project root (where tuner_example.py lives) ───────────────────
cd "${SLURM_SUBMIT_DIR}"

# ── (4) Launch tuner_example.py ───────────────────────────────────────────────
# tuner_example.py internally defines the 10 Atari v5 games and their normalization.
# It expects a CleanRL EnvPool entry point at --script and forwards --optimizer and --wandb-tag.
echo "[info] Starting tuner_example.py with optimizer=${OPTIMIZER} and wandb_tag=${WANDB_TAG}"
srun python -u tuner_example.py \
  --script "${SCRIPT}" \
  --gpus "${GPUS}" \
  --optimizer "${OPTIMIZER}" \
  --trials ${TRIALS} \
  --seeds ${SEEDS} \
  --study-name "${STUDY_NAME}" \
  --storage "${STORAGE_URL}" \
  --logs-root "${LOGS_ROOT}" \
  --wandb-tag "${WANDB_TAG}"

echo "Job finished on $(date)"
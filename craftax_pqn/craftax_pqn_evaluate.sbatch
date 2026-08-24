#!/bin/bash
#SBATCH --job-name=craftax-pqn-eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/craftax/slurm/%x-%j.out
#SBATCH --error=logs/craftax/slurm/%x-%j.err
#SBATCH --account=aalaeddini_391_0001

set -euo pipefail
trap 'code=$?; echo "[fatal] exit=${code} line=${LINENO} date=$(date)"; exit "${code}"' ERR

ENVS="Craftax-Classic-Symbolic-v1"
NUM_SEEDS=10
SEED_START=101
TOTAL_TIMESTEPS=1000000
EVAL_STEPS=200000
EVAL_NUM_ENVS=32
OUTPUT_ROOT="logs/craftax/evaluation"
LOGS_ROOT="logs/craftax/tuning"
ADAM_STUDY="craftax_pqn_adam"
MUON_STUDY="craftax_pqn_muon"
CPUS_PER_RUN=8

TRACK=1
WANDB_PROJECT="cleanRL-craftax-pqn-evaluation"
WANDB_ENTITY=""
WANDB_GROUP="craftax-pqn-final-evaluation"

cd "${SLURM_SUBMIT_DIR}"
module purge
module load conda
module load miniforge/24.11.2-1
source ~/.bashrc
conda activate RLMuon

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35

mkdir -p "${OUTPUT_ROOT}" logs/craftax/slurm
ADAM_CONFIG="${LOGS_ROOT}/${ADAM_STUDY}/best_hyperparams.json"
MUON_CONFIG="${LOGS_ROOT}/${MUON_STUDY}/best_hyperparams.json"
test -f "${ADAM_CONFIG}"
test -f "${MUON_CONFIG}"

GPU_COUNT=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
if [[ "${GPU_COUNT}" -le 0 ]]; then
    echo "[fatal] No CUDA devices are visible inside the Slurm allocation."
    exit 1
fi
GPUS=$(python - <<'PY'
import torch
print(",".join(str(index) for index in range(torch.cuda.device_count())))
PY
)

TRACK_ARGS=(--no-track)
if [[ "${TRACK}" == "1" ]]; then
    TRACK_ARGS=(
        --track
        --wandb-project-name "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
    )
    if [[ -n "${WANDB_ENTITY}" ]]; then
        TRACK_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
    fi
fi

srun python -u craftax_pqn/evaluate_craftax.py \
    --configs "${ADAM_CONFIG}" "${MUON_CONFIG}" \
    --envs "${ENVS}" \
    --gpus "${GPUS}" \
    --num-seeds "${NUM_SEEDS}" \
    --seed-start "${SEED_START}" \
    --output-root "${OUTPUT_ROOT}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-steps "${EVAL_STEPS}" \
    --eval-num-envs "${EVAL_NUM_ENVS}" \
    --cpus-per-run "${CPUS_PER_RUN}" \
    "${TRACK_ARGS[@]}"

test -f "${OUTPUT_ROOT}/run_results.csv"
test -f "${OUTPUT_ROOT}/environment_summary.csv"
test -f "${OUTPUT_ROOT}/overall_summary.csv"
test -f "${OUTPUT_ROOT}/paired_deltas.csv"
echo "[info] Craftax evaluation complete at $(date)"

#!/bin/bash
#SBATCH --job-name=deepsea-pqn-eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/deepsea/slurm/%x-%j.out
#SBATCH --error=logs/deepsea/slurm/%x-%j.err
#SBATCH --account=aalaeddini_391_0001

set -euo pipefail
trap 'code=$?; echo "[fatal] exit=${code} line=${LINENO} date=$(date)"; exit "${code}"' ERR

SIZES="10,20,30,40,50"
NUM_SEEDS=10
SEED_START=101
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=4096
OUTPUT_ROOT="logs/deepsea/evaluation"
LOGS_ROOT="logs/deepsea/tuning"
ADAM_STUDY="deepsea_pqn_adam"
MUON_STUDY="deepsea_pqn_muon"
CPUS_PER_RUN=8

TRACK=1
WANDB_PROJECT="cleanRL-deepsea-pqn-evaluation"
WANDB_ENTITY=""
WANDB_GROUP="deepsea-pqn-final-evaluation"

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

mkdir -p "${OUTPUT_ROOT}" logs/deepsea/slurm
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

srun python -u deepsea/evaluate_deepsea.py \
    --configs "${ADAM_CONFIG}" "${MUON_CONFIG}" \
    --sizes "${SIZES}" \
    --gpus "${GPUS}" \
    --num-seeds "${NUM_SEEDS}" \
    --seed-start "${SEED_START}" \
    --output-root "${OUTPUT_ROOT}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --cpus-per-run "${CPUS_PER_RUN}" \
    "${TRACK_ARGS[@]}"

test -f "${OUTPUT_ROOT}/run_results.csv"
test -f "${OUTPUT_ROOT}/size_summary.csv"
test -f "${OUTPUT_ROOT}/overall_summary.csv"
test -f "${OUTPUT_ROOT}/paired_deltas.csv"
echo "[info] DeepSea evaluation complete at $(date)"

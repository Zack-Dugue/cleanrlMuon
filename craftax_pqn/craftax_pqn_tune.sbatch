#!/bin/bash
#SBATCH --job-name=craftax-pqn-tune
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
TRIALS=30
SEEDS_PER_TRIAL=3
SEED_START=1
SEARCH_SPACE="core"
TOTAL_TIMESTEPS=1000000
NUM_ENVS=128
NUM_STEPS=64
NUM_MINIBATCHES=4
EVAL_STEPS=100000
EVAL_NUM_ENVS=32
LOGS_ROOT="logs/craftax/tuning"
ADAM_STUDY="craftax_pqn_adam"
MUON_STUDY="craftax_pqn_muon"
CPUS_PER_RUN=8

TRACK=1
WANDB_PROJECT="cleanRL-craftax-pqn-tuning"
WANDB_ENTITY=""

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

mkdir -p "${LOGS_ROOT}" logs/craftax/slurm
test -f craftax_pqn/tune_craftax.py
test -f cleanrl/pqn_craftax.py

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
    TRACK_ARGS=(--track --wandb-project-name "${WANDB_PROJECT}")
    if [[ -n "${WANDB_ENTITY}" ]]; then
        TRACK_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
    fi
fi

run_study() {
    local optimizer="$1"
    local study_name="$2"
    echo "[info] optimizer=${optimizer} study=${study_name} GPUs=${GPUS} date=$(date)"
    srun python -u craftax_pqn/tune_craftax.py \
        --optimizer "${optimizer}" \
        --envs "${ENVS}" \
        --gpus "${GPUS}" \
        --trials "${TRIALS}" \
        --seeds "${SEEDS_PER_TRIAL}" \
        --seed-start "${SEED_START}" \
        --study-name "${study_name}" \
        --search-space "${SEARCH_SPACE}" \
        --logs-root "${LOGS_ROOT}" \
        --total-timesteps "${TOTAL_TIMESTEPS}" \
        --num-envs "${NUM_ENVS}" \
        --num-steps "${NUM_STEPS}" \
        --num-minibatches "${NUM_MINIBATCHES}" \
        --eval-steps "${EVAL_STEPS}" \
        --eval-num-envs "${EVAL_NUM_ENVS}" \
        --cpus-per-run "${CPUS_PER_RUN}" \
        "${TRACK_ARGS[@]}"
}

run_study Adam "${ADAM_STUDY}"
run_study Muon "${MUON_STUDY}"

test -f "${LOGS_ROOT}/${ADAM_STUDY}/best_hyperparams.json"
test -f "${LOGS_ROOT}/${MUON_STUDY}/best_hyperparams.json"
echo "[info] Craftax tuning complete at $(date)"

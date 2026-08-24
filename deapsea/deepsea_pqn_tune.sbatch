#!/bin/bash
#SBATCH --job-name=deepsea-pqn-tune
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

SIZES="10,20,30"
TRIALS=30
SEEDS_PER_TRIAL=3
SEED_START=1
SEARCH_SPACE="core"
TOTAL_TIMESTEPS=1000000
NUM_ENVS=2048
NUM_STEPS=32
NUM_MINIBATCHES=4
EVAL_EPISODES=2048
LOGS_ROOT="logs/deepsea/tuning"
ADAM_STUDY="deepsea_pqn_adam"
MUON_STUDY="deepsea_pqn_muon"
CPUS_PER_RUN=8

# W&B is on by default for the spawned trainers. Set TRACK=0 to disable it.
TRACK=1
WANDB_PROJECT="cleanRL-deepsea-pqn-tuning"
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

mkdir -p "${LOGS_ROOT}" logs/deepsea/slurm
test -f deepsea/tune_deepsea.py
test -f cleanrl/pqn_deepsea.py

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
    srun python -u deepsea/tune_deepsea.py \
        --optimizer "${optimizer}" \
        --sizes "${SIZES}" \
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
        --eval-episodes "${EVAL_EPISODES}" \
        --cpus-per-run "${CPUS_PER_RUN}" \
        "${TRACK_ARGS[@]}"
}

# Each optimizer gets all allocated GPUs and its own learning-rate study.
run_study Adam "${ADAM_STUDY}"
run_study Muon "${MUON_STUDY}"

test -f "${LOGS_ROOT}/${ADAM_STUDY}/best_hyperparams.json"
test -f "${LOGS_ROOT}/${MUON_STUDY}/best_hyperparams.json"
echo "[info] DeepSea tuning complete at $(date)"

#!/bin/bash
#SBATCH --job-name=mujoco-pqn-tune
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=8
#SBATCH --time=48:00:00
#SBATCH --output=mujoco-pqn-tune-%j.out
#SBATCH --error=mujoco-pqn-tune-%j.err
#SBATCH --account=aalaeddini_391_0001

set -euo pipefail

trap 'code=$?; echo "[fatal] exit=${code} line=${LINENO} date=$(date)"; exit "${code}"' ERR

# ── Experiment settings ──────────────────────────────────────────────────────
ENVIRONMENTS="HalfCheetah-v4,Hopper-v4,Walker2d-v4"
TRIALS=30
SEEDS_PER_TRIAL=2
SEED_START=1
SEARCH_SPACE="core"

TOTAL_TIMESTEPS=500000
NUM_ENVS=16
NUM_STEPS=32
NUM_MINIBATCHES=4
EVAL_EPISODES=5

LOGS_ROOT="mujoco/tuner_logs"
ADAM_STUDY="mujoco_pqn_adam"
MUON_STUDY="mujoco_pqn_muon"
CPUS_PER_RUN=8

# Set to 1 to enable W&B for every spawned training process.
TRACK=0
WANDB_PROJECT="cleanRL-mujoco-pqn-tuning"
WANDB_ENTITY=""

# ── Environment setup ────────────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR}"
module purge
module load miniforge/24.11.2-1
source ~/.bashrc
conda activate RLMuon

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${LOGS_ROOT}"

if [[ ! -f mujoco/tune_mujoco.py || ! -f cleanrl/pqn_mujoco.py ]]; then
    echo "[fatal] Submit this job from the cleanrlMuon repository root."
    exit 1
fi

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

echo "[info] date=$(date) host=$(hostname) job=${SLURM_JOB_ID}"
echo "[info] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[info] local GPU ids=${GPUS}"
echo "[info] environments=${ENVIRONMENTS} trials=${TRIALS} seeds=${SEEDS_PER_TRIAL}"

TRACK_ARGS=()
if [[ "${TRACK}" == "1" ]]; then
    TRACK_ARGS+=(--track --wandb-project-name "${WANDB_PROJECT}")
    if [[ -n "${WANDB_ENTITY}" ]]; then
        TRACK_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
    fi
fi

run_study() {
    local optimizer="$1"
    local study_name="$2"

    echo "[info] starting optimizer=${optimizer} study=${study_name} at $(date)"
    srun python -u mujoco/tune_mujoco.py \
        --optimizer "${optimizer}" \
        --envs "${ENVIRONMENTS}" \
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

# Each study uses all allocated GPUs. Keeping them sequential avoids two studies
# oversubscribing the same devices and makes optimizer-specific failures clear.
run_study "Adam" "${ADAM_STUDY}"
run_study "Muon" "${MUON_STUDY}"

ADAM_CONFIG="${LOGS_ROOT}/${ADAM_STUDY}/best_hyperparams.json"
MUON_CONFIG="${LOGS_ROOT}/${MUON_STUDY}/best_hyperparams.json"
test -f "${ADAM_CONFIG}"
test -f "${MUON_CONFIG}"

echo "[info] tuning complete at $(date)"
echo "[info] Adam config: ${ADAM_CONFIG}"
echo "[info] Muon config: ${MUON_CONFIG}"

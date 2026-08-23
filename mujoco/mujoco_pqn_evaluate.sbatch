#!/bin/bash
#SBATCH --job-name=mujoco-pqn-eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --output=mujoco-pqn-eval-%j.out
#SBATCH --error=mujoco-pqn-eval-%j.err
#SBATCH --account=aalaeddini_391_0001

set -euo pipefail

trap 'code=$?; echo "[fatal] exit=${code} line=${LINENO} date=$(date)"; exit "${code}"' ERR

# ── Experiment settings ──────────────────────────────────────────────────────
TUNER_LOGS_ROOT="mujoco/tuner_logs"
ADAM_CONFIG="${TUNER_LOGS_ROOT}/mujoco_pqn_adam/best_hyperparams.json"
MUON_CONFIG="${TUNER_LOGS_ROOT}/mujoco_pqn_muon/best_hyperparams.json"

ENVIRONMENTS="HalfCheetah-v4,Hopper-v4,Walker2d-v4,Ant-v4"
NUM_SEEDS=10
SEED_START=101
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=10
GREEDY_EVAL_STEPS=50000
GREEDY_EVAL_NUM_ENVS=8

OUTPUT_ROOT="mujoco/evaluation"
CPUS_PER_RUN=8

# Set to 1 to enable W&B for every spawned training process.
TRACK=0
WANDB_PROJECT="cleanRL-mujoco-pqn-evaluation"
WANDB_ENTITY=""
WANDB_GROUP="mujoco-pqn-final-evaluation"

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

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f mujoco/evaluate_mujoco.py || ! -f cleanrl/pqn_mujoco.py ]]; then
    echo "[fatal] Submit this job from the cleanrlMuon repository root."
    exit 1
fi
for config in "${ADAM_CONFIG}" "${MUON_CONFIG}"; do
    if [[ ! -f "${config}" ]]; then
        echo "[fatal] Missing tuned configuration: ${config}"
        exit 1
    fi
done

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
echo "[info] environments=${ENVIRONMENTS} seeds=${NUM_SEEDS}"
echo "[info] greedy evaluation transitions=${GREEDY_EVAL_STEPS}"

TRACK_ARGS=()
if [[ "${TRACK}" == "1" ]]; then
    TRACK_ARGS+=(
        --track
        --wandb-project-name "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
    )
    if [[ -n "${WANDB_ENTITY}" ]]; then
        TRACK_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
    fi
fi

srun python -u mujoco/evaluate_mujoco.py \
    --configs "${ADAM_CONFIG}" "${MUON_CONFIG}" \
    --envs "${ENVIRONMENTS}" \
    --gpus "${GPUS}" \
    --num-seeds "${NUM_SEEDS}" \
    --seed-start "${SEED_START}" \
    --output-root "${OUTPUT_ROOT}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --greedy-eval-steps "${GREEDY_EVAL_STEPS}" \
    --greedy-eval-num-envs "${GREEDY_EVAL_NUM_ENVS}" \
    --cpus-per-run "${CPUS_PER_RUN}" \
    "${TRACK_ARGS[@]}"

echo "[info] evaluation complete at $(date)"
echo "[info] results: ${OUTPUT_ROOT}/overall_summary.csv"

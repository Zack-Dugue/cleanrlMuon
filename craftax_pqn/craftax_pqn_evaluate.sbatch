#!/bin/bash
#SBATCH --job-name=craftax-pqn-eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=8
#SBATCH --time=36:00:00
#SBATCH --output=logs/craftax/slurm/%x-%j.out
#SBATCH --error=logs/craftax/slurm/%x-%j.err
#SBATCH --account=aalaeddini_391_0003

set -euo pipefail
trap 'code=$?; echo "[fatal] exit=${code} line=${LINENO} date=$(date)"; exit "${code}"' ERR

ENVS="Craftax-Symbolic-v1"
NUM_SEEDS=10
SEED_START=101
TOTAL_TIMESTEPS=1000000000
EVAL_STEPS=10000
EVAL_NUM_ENVS=512
WEIGHT_DECAY=0.0001
MATRIX_DIAGNOSTICS_INTERVAL=100
MATRIX_DIAGNOSTICS_POWER_ITERATIONS=8
OUTPUT_ROOT="logs/craftax/evaluation_1b"
LOGS_ROOT="logs/craftax/tuning"
ADAM_STUDY="craftax_pqn_rnn_adam"
MUON_STUDY="craftax_pqn_rnn_muon"
CPUS_PER_RUN=8
CONDA_ENV="RLMuonJAX"

TRACK=1
WANDB_PROJECT="cleanRL-craftax-pqn-1b"
WANDB_ENTITY=""
WANDB_GROUP="craftax-pqn-final-evaluation"

cd "${SLURM_SUBMIT_DIR}"
module purge
module load conda
module load miniforge/24.11.2-1
source ~/.bashrc
conda activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

mkdir -p "${OUTPUT_ROOT}" logs/craftax/slurm
ADAM_CONFIG="${LOGS_ROOT}/${ADAM_STUDY}/best_hyperparams.json"
MUON_CONFIG="${LOGS_ROOT}/${MUON_STUDY}/best_hyperparams.json"
test -f "${ADAM_CONFIG}"
test -f "${MUON_CONFIG}"

GPU_COUNT=$(python - <<'PY'
import jax
print(len(jax.devices("gpu")))
PY
)
if [[ "${GPU_COUNT}" -le 0 ]]; then
    echo "[fatal] No CUDA devices are visible inside the Slurm allocation."
    exit 1
fi
GPUS=$(python - <<'PY'
import jax
print(",".join(str(index) for index in range(len(jax.devices("gpu")))))
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
    --configs "${MUON_CONFIG}" "${ADAM_CONFIG}" \
    --envs "${ENVS}" \
    --gpus "${GPUS}" \
    --num-seeds "${NUM_SEEDS}" \
    --seed-start "${SEED_START}" \
    --output-root "${OUTPUT_ROOT}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-steps "${EVAL_STEPS}" \
    --eval-num-envs "${EVAL_NUM_ENVS}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --matrix-diagnostics-interval "${MATRIX_DIAGNOSTICS_INTERVAL}" \
    --matrix-diagnostics-power-iterations "${MATRIX_DIAGNOSTICS_POWER_ITERATIONS}" \
    --scale-batch-renorm-warmup \
    --sequential-configs \
    --cpus-per-run "${CPUS_PER_RUN}" \
    "${TRACK_ARGS[@]}"

test -f "${OUTPUT_ROOT}/run_results.csv"
test -f "${OUTPUT_ROOT}/environment_summary.csv"
test -f "${OUTPUT_ROOT}/overall_summary.csv"
test -f "${OUTPUT_ROOT}/paired_deltas.csv"
test -f "${OUTPUT_ROOT}/resolved_configs.json"
echo "[info] Craftax evaluation complete at $(date)"

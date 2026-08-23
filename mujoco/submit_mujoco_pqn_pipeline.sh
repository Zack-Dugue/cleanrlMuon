#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

TUNE_JOB_ID=$(sbatch --parsable mujoco/mujoco_pqn_tune.sbatch)
EVAL_JOB_ID=$(sbatch \
    --parsable \
    --dependency="afterok:${TUNE_JOB_ID}" \
    mujoco/mujoco_pqn_evaluate.sbatch)

echo "Submitted tuning job: ${TUNE_JOB_ID}"
echo "Submitted evaluation job: ${EVAL_JOB_ID}"
echo "Evaluation will start only after tuning exits successfully."

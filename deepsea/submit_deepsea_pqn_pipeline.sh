#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# Slurm opens output files before the job script itself runs.
mkdir -p logs/deepsea/slurm

TUNE_JOB_ID=$(sbatch --parsable deepsea/deepsea_pqn_tune.sbatch)
EVAL_JOB_ID=$(sbatch \
    --parsable \
    --dependency="afterok:${TUNE_JOB_ID}" \
    deepsea/deepsea_pqn_evaluate.sbatch)

echo "Submitted DeepSea tuning job: ${TUNE_JOB_ID}"
echo "Submitted DeepSea evaluation job: ${EVAL_JOB_ID}"
echo "Evaluation will start only after tuning exits successfully."

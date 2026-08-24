#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

mkdir -p logs/craftax/slurm

TUNE_JOB_ID=$(sbatch --parsable craftax_pqn/craftax_pqn_tune.sbatch)
EVAL_JOB_ID=$(sbatch \
    --parsable \
    --dependency="afterok:${TUNE_JOB_ID}" \
    craftax_pqn/craftax_pqn_evaluate.sbatch)

echo "Submitted Craftax tuning job: ${TUNE_JOB_ID}"
echo "Submitted Craftax evaluation job: ${EVAL_JOB_ID}"
echo "Evaluation will start only after tuning exits successfully."

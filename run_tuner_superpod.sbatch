#!/bin/bash
#
# file: run_tuner_superpod.sbatch
#
#SBATCH --job-name=ppo-tune
#SBATCH --partition=batch            # or 'short' (4h limit)
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # one launcher; Python does per-GPU workers
#SBATCH --gpus=8                     # change to 1..8 as needed
#SBATCH --time=48:00:00              # must fit the partition limit (batch=48h, short=4h)
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Optional: if SMU uses accounts/QOS, uncomment and set:
# SBATCH --account=<your_account>
# SBATCH --qos=<your_qos>

###############################################################################
# Environment setup (modules/conda). Adjust to your environment on SuperPOD.
###############################################################################
set -euo pipefail

# Slurm creates $SLURM_SUBMIT_DIR; cd there so relative paths work
cd "$SLURM_SUBMIT_DIR"

# Make sure log dir exists (matches #SBATCH lines above)
mkdir -p logs

# If your site uses Lmod modules, you can load CUDA/PyTorch here.
# Many users just rely on conda; if so, source your shell init and activate env:
# (Use bash on SMU login/compute nodes.)
source ~/.bashrc
conda activate <your_env>

# Sanity: print CUDA devices visible to the job
echo "[info] CUDA visible devices: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[info] torch.cuda.device_count() =', torch.cuda.device_count())"

###############################################################################
# Recommended single-node NCCL / OpenMP settings (safe defaults)
###############################################################################
# Let PyTorch/NCCL prefer faster intra-node transport; SuperPOD is NVLink/NVSwitch.
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1         # single-node, prefer NVLink/NVSwitch
export OMP_NUM_THREADS=4         # each worker will also set OMP threads if you coded it

###############################################################################
# W&B (optional)
###############################################################################
# If you want runs to sync to your W&B account, either login once on the cluster
# or export the API key here. Leave commented if you prefer offline logging.
# export WANDB_API_KEY=<your_wandb_api_key>
# WandB project/entity can also be set via CLI to your tuner; we also export envs:
export WANDB_PROJECT=cleanRL
export WANDB_ENTITY=<your_wandb_entity>   # or comment out

###############################################################################
# Launch your tuner
###############################################################################
# Tune momentum + lr/ent_coef/update_epochs as you set in the Python.
# --num_trials / --num_seeds are up to you; start small to shake out issues.
# --wandb_enable True will pass W&B env through; your Python avoids boolean Tyro flags.
# Example with 8 GPUs on a single node (this script requests 8 GPUs above):
RUN_TS=$(date +%Y%m%d_%H%M%S)

python -u multi_env_tuner_atari_dp.py \
  --num_trials 10 \
  --num_seeds 3 \
  --study_name "ppo_atari_superpod_${RUN_TS}" \
  --results_dir "tuning_results" \
  --wandb_enable True \
  --wandb_project "cleanRL" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  > "logs/launcher_${SLURM_JOB_ID}.log" 2>&1

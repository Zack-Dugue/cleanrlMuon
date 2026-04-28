#!/bin/bash
#
#SBATCH --job-name=cleanrl-montezuma-video
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=64
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=aalaeddini_391_0001
# If your cluster requires a QoS line, uncomment and set it:
# #SBATCH --qos=YOUR_QOS_NAME
# Optional memory line:
# #SBATCH --mem=64G

set -euo pipefail

# ── (0) Config ────────────────────────────────────────────────────────────────
ENV_ID="MontezumaRevenge-v5"
TOTAL_TIMESTEPS=10000000
NUM_ENVS=64
SEED=1

WANDB_PROJECT="cleanrl-montezuma-rnd-comparison"
WANDB_TAG="montezuma_ppo_vs_rnd"

RND_SCRIPT_ORIG="cleanrl/ppo_rnd_envpool.py"
PPO_SCRIPT_ORIG="cleanrl/ppo_atari_envpool.py"

RND_SCRIPT_PATCHED="patched_cleanrl/ppo_rnd_envpool_final_video.py"
PPO_SCRIPT_PATCHED="patched_cleanrl/ppo_atari_envpool_final_video.py"

# ── (1) Basic runtime setup ───────────────────────────────────────────────────
mkdir -p logs checkpoints videos patched_cleanrl

echo "Job started on $(date) at host $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"(assigned by Slurm)"}"

# ===== Activate your environment =====
module purge
module load conda
source ~/.bashrc
conda init
conda activate RLMuon

# Your tuner script also loaded this later, so keep it here if your site needs it.
module load miniforge/24.11.2-1
conda activate RLMuon

cd "${SLURM_SUBMIT_DIR}"

# Runtime env vars
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_CPP_MIN_LOG_LEVEL=2
export LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Sanity checks
echo "[info] Host: $(hostname)"
echo "[info] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"(unset)"}"
echo "[info] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
python - <<'PY'
import torch, os
print("[info] torch.cuda.device_count() =", torch.cuda.device_count())
print("[info] GPUs visible:", list(range(torch.cuda.device_count())))
print("[info] OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
print("[info] MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
PY

# ── (2) Copy and patch CleanRL scripts ────────────────────────────────────────
cp "${RND_SCRIPT_ORIG}" "${RND_SCRIPT_PATCHED}"
cp "${PPO_SCRIPT_ORIG}" "${PPO_SCRIPT_PATCHED}"

python - <<'PY'
from pathlib import Path

PATCH = r'''
# ---- final checkpoint + final evaluation video patch ----
import pathlib
import imageio.v2 as imageio

pathlib.Path("checkpoints").mkdir(exist_ok=True)
pathlib.Path("videos").mkdir(exist_ok=True)

ckpt_path = f"checkpoints/{run_name}.agent.pt"
torch.save(agent.state_dict(), ckpt_path)
print(f"saved final agent checkpoint to {ckpt_path}")

video_path = f"videos/{run_name}.final.mp4"
agent.eval()

eval_envs = envpool.make(
    args.env_id,
    env_type="gym",
    num_envs=1,
    episodic_life=False,
    reward_clip=False,
    seed=args.seed + 10000,
)
eval_envs.num_envs = 1
eval_envs.single_action_space = eval_envs.action_space
eval_envs.single_observation_space = eval_envs.observation_space

eval_obs = torch.Tensor(eval_envs.reset()).to(device)
frames = []
episode_return = 0.0

# Montezuma episodes can be long; this caps video length.
for _eval_step in range(27000):
    # Save the most recent frame from the 4-frame stack.
    frame = eval_obs[0, -1].detach().cpu().numpy()
    frames.append(frame.astype("uint8"))

    with torch.no_grad():
        hidden = agent.network(eval_obs / 255.0)
        logits = agent.actor(hidden)
        action = torch.argmax(logits, dim=1)

    next_obs, reward, done, info = eval_envs.step(action.cpu().numpy())
    episode_return += float(reward[0])
    eval_obs = torch.Tensor(next_obs).to(device)

    if bool(done[0]):
        break

eval_envs.close()

if frames:
    imageio.mimsave(video_path, frames, fps=30)
    print(f"saved final evaluation video to {video_path}")
    print(f"final evaluation return: {episode_return}")

    if args.track:
        wandb.log({
            "final_eval_video": wandb.Video(video_path, fps=30, format="mp4"),
            "final_eval_return": episode_return,
        })
else:
    print("warning: no frames captured for final evaluation video")
# ---- end patch ----
'''

for path in [
    Path("patched_cleanrl/ppo_rnd_envpool_final_video.py"),
    Path("patched_cleanrl/ppo_atari_envpool_final_video.py"),
]:
    text = path.read_text()
    needle = "envs.close()\n    writer.close()"
    if needle not in text:
        raise RuntimeError(f"Could not find patch point in {path}")
    text = text.replace(needle, PATCH + "\n    envs.close()\n    writer.close()")
    path.write_text(text)

print("Patched CleanRL scripts written to patched_cleanrl/")
PY

# ── (3) Launch PPO+RND and PPO in parallel ────────────────────────────────────
echo "[info] Starting PPO+RND and normal PPO on ${ENV_ID}"
echo "[info] TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
echo "[info] NUM_ENVS per run=${NUM_ENVS}"
echo "[info] Each run gets 1 GPU and 64 CPUs"

srun \
  --exclusive \
  --ntasks=1 \
  --cpus-per-task=64 \
  --gpus-per-task=1 \
  python -u "${RND_SCRIPT_PATCHED}" \
    --env-id "${ENV_ID}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --num-envs "${NUM_ENVS}" \
    --seed "${SEED}" \
    --track \
    --wandb-project-name "${WANDB_PROJECT}" \
    --wandb-tag "${WANDB_TAG}" &

srun \
  --exclusive \
  --ntasks=1 \
  --cpus-per-task=64 \
  --gpus-per-task=1 \
  python -u "${PPO_SCRIPT_PATCHED}" \
    --env-id "${ENV_ID}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --num-envs "${NUM_ENVS}" \
    --seed "${SEED}" \
    --track \
    --wandb-project-name "${WANDB_PROJECT}" \
    --wandb-tag "${WANDB_TAG}" &

wait

echo "Both runs completed."
echo "Final checkpoints are in ./checkpoints"
echo "Final videos are in ./videos"
echo "Job finished on $(date)"
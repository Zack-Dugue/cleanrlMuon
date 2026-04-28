#!/bin/bash
#SBATCH --job-name=cleanrl-montezuma-video
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs checkpoints videos patched_cleanrl

# Optional: load modules / activate env here
# module load cuda
# source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

ENV_ID="MontezumaRevenge-v5"
TOTAL_TIMESTEPS=10000000
NUM_ENVS=64
SEED=1

# Copy scripts so the original CleanRL files stay untouched.
cp cleanrl/ppo_rnd_envpool.py patched_cleanrl/ppo_rnd_envpool_final_video.py
cp cleanrl/ppo_atari_envpool.py patched_cleanrl/ppo_atari_envpool_final_video.py

# Patch both scripts:
# - save the trained agent checkpoint
# - record a final greedy rollout as an mp4 from the stacked 84x84 Atari observations
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

echo "Starting PPO+RND and normal PPO on ${ENV_ID}"
echo "Job ID: ${SLURM_JOB_ID}"

srun \
  --exclusive \
  --ntasks=1 \
  --cpus-per-task=64 \
  --gres=gpu:1 \
  bash -c '
    export CUDA_VISIBLE_DEVICES=0
    echo "Starting PPO+RND on assigned GPU ${CUDA_VISIBLE_DEVICES}"
    uv run python patched_cleanrl/ppo_rnd_envpool_final_video.py \
      --env-id '"${ENV_ID}"' \
      --total-timesteps '"${TOTAL_TIMESTEPS}"' \
      --num-envs '"${NUM_ENVS}"' \
      --seed '"${SEED}"' \
      --track
  ' &

srun \
  --exclusive \
  --ntasks=1 \
  --cpus-per-task=64 \
  --gres=gpu:1 \
  bash -c '
    export CUDA_VISIBLE_DEVICES=0
    echo "Starting normal PPO EnvPool on assigned GPU ${CUDA_VISIBLE_DEVICES}"
    uv run python patched_cleanrl/ppo_atari_envpool_final_video.py \
      --env-id '"${ENV_ID}"' \
      --total-timesteps '"${TOTAL_TIMESTEPS}"' \
      --num-envs '"${NUM_ENVS}"' \
      --seed '"${SEED}"' \
      --track
  ' &

wait

echo "Both runs completed."
echo "Final checkpoints are in ./checkpoints"
echo "Final videos are in ./videos"
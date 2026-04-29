#!/usr/bin/env python3
"""
Record EnvPool grayscale Atari videos with RND value-gradient saliency overlays.

Overlay meaning:
  RED  = squared input-gradient saliency for V_ext
  BLUE = squared input-gradient saliency for V_int
  MAGENTA = both V_ext and V_int are sensitive there

This intentionally records the agent's ugly-but-faithful 84x84 grayscale input,
not RGB rendered Atari frames.

Example:

python record_checkpoint_video.py \
  --checkpoint checkpoints/MontezumaRevenge-v5__ppo_rnd_envpool_final_video__1__1777366098.agent.pt \
  --env-id MontezumaRevenge-v5 \
  --output videos/rnd_saliency_target500.mp4 \
  --cuda \
  --target-return 500 \
  --max-attempts 100 \
  --stochastic
"""

import argparse
import glob
from pathlib import Path

import envpool
import gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Matches your patched RND / PPO EnvPool agent architecture.

    RND checkpoint has:
      network, extra_layer, actor, critic_ext, critic_int

    PPO checkpoint has:
      network, extra_layer, actor, critic_ext

    For this saliency script, critic_int is required if you want blue V_int overlay.
    """

    def __init__(self, envs, use_rnd_critic: bool):
        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.LayerNorm(256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.LayerNorm(448),
            nn.ReLU(),
        )

        self.extra_layer = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.1),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.LayerNorm(448),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )

        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)

        if use_rnd_critic:
            self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def actor_logits(self, x):
        hidden = self.network(x / 255.0)
        return self.actor(hidden)

    def get_action(self, x, deterministic=True):
        logits = self.actor_logits(x)
        if deterministic:
            return torch.argmax(logits, dim=1)
        probs = Categorical(logits=logits)
        return probs.sample()

    def get_values(self, x):
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        fused = features + hidden

        v_ext = self.critic_ext(fused)

        if hasattr(self, "critic_int"):
            v_int = self.critic_int(fused)
        else:
            v_int = None

        return v_ext, v_int


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_is_rnd_checkpoint(state_dict):
    return any(k.startswith("critic_int.") for k in state_dict.keys())


def resolve_checkpoint(path_or_glob):
    path = Path(path_or_glob)
    if path.exists():
        return path

    matches = sorted(glob.glob(path_or_glob))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for: {path_or_glob}")

    if len(matches) > 1:
        print("[warning] Multiple checkpoints matched. Using first:")
        for m in matches:
            print("  ", m)

    return Path(matches[0])


def make_env(env_id, seed, repeat_action_probability):
    envs = envpool.make(
        env_id,
        env_type="gym",
        num_envs=1,
        episodic_life=False,
        reward_clip=False,
        seed=seed,
        repeat_action_probability=repeat_action_probability,
    )

    envs.num_envs = 1
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    assert isinstance(envs.action_space, gym.spaces.Discrete), "Only discrete Atari action spaces supported."
    return envs


def normalize_heatmap(x, eps=1e-8, percentile=99.5):
    """
    Per-frame robust normalization to [0, 1].
    This makes the saliency visible even when raw gradients are tiny.
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, 0.0)

    hi = np.percentile(x, percentile)
    if hi < eps:
        hi = float(x.max())

    if hi < eps:
        return np.zeros_like(x, dtype=np.float32)

    return np.clip(x / (hi + eps), 0.0, 1.0)


def make_saliency_overlay_frame(
    obs_tensor,
    ext_saliency,
    int_saliency,
    alpha=0.75,
    upscale=4,
):
    """
    obs_tensor: [1, 4, 84, 84]
    ext_saliency: [84, 84], red
    int_saliency: [84, 84], blue

    Returns RGB uint8 frame.
    """
    gray = obs_tensor[0, -1].detach().cpu().numpy().astype(np.float32)
    gray = np.clip(gray, 0, 255) / 255.0

    ext_h = normalize_heatmap(ext_saliency)
    int_h = normalize_heatmap(int_saliency)

    rgb = np.repeat(gray[..., None], 3, axis=2)

    # Add red for V_ext.
    rgb[..., 0] = np.clip((1.0 - alpha * ext_h) * rgb[..., 0] + alpha * ext_h, 0.0, 1.0)

    # Add blue for V_int.
    rgb[..., 2] = np.clip((1.0 - alpha * int_h) * rgb[..., 2] + alpha * int_h, 0.0, 1.0)

    # Slightly suppress green where either saliency is large, so red/blue are clearer.
    combined = np.maximum(ext_h, int_h)
    rgb[..., 1] = np.clip((1.0 - 0.35 * combined) * rgb[..., 1], 0.0, 1.0)

    frame = (rgb * 255.0).astype(np.uint8)

    if upscale and upscale > 1:
        frame = np.repeat(np.repeat(frame, upscale, axis=0), upscale, axis=1)

    return frame


def compute_value_saliency(agent, obs_tensor):
    """
    Computes squared input gradients:

      sal_ext = sum_channels( dV_ext/dobs ^ 2 )
      sal_int = sum_channels( dV_int/dobs ^ 2 )

    Returns saliency maps shaped [84, 84].

    Important:
      We sum over all 4 stacked frames to show which pixels across the input
      stack affect the value. The displayed image is still the latest frame.
    """
    agent.zero_grad(set_to_none=True)

    x = obs_tensor.detach().clone().float()
    x.requires_grad_(True)

    v_ext, v_int = agent.get_values(x)

    # V_ext saliency
    if x.grad is not None:
        x.grad.zero_()
    v_ext.sum().backward(retain_graph=True)
    grad_ext = x.grad.detach().clone()

    sal_ext = grad_ext.pow(2).sum(dim=1)[0].detach().cpu().numpy()

    # V_int saliency
    if v_int is None:
        sal_int = np.zeros_like(sal_ext, dtype=np.float32)
    else:
        x.grad.zero_()
        v_int.sum().backward()
        grad_int = x.grad.detach().clone()
        sal_int = grad_int.pow(2).sum(dim=1)[0].detach().cpu().numpy()

    return sal_ext, sal_int, float(v_ext.item()), None if v_int is None else float(v_int.item())


def add_top_bar(frame, episode_return, step, v_ext, v_int, success=False):
    """
    Adds a simple colored top strip.
    Avoids requiring cv2/PIL/text libraries.

    Bar encoding:
      left red intensity   roughly tracks max(0, V_ext)
      right blue intensity roughly tracks max(0, V_int)
      green strip at very top if this is the kept/success episode
    """
    frame = frame.copy()
    h, w, _ = frame.shape

    bar_h = max(8, h // 24)

    # Dark bar background
    frame[:bar_h, :, :] = (frame[:bar_h, :, :] * 0.35).astype(np.uint8)

    # Crude value bars. Tanh keeps them bounded.
    ext_mag = float(np.tanh(abs(v_ext) / 10.0)) if v_ext is not None else 0.0
    int_mag = float(np.tanh(abs(v_int) / 10.0)) if v_int is not None else 0.0

    ext_w = int((w // 2) * ext_mag)
    int_w = int((w // 2) * int_mag)

    if ext_w > 0:
        frame[1:bar_h - 1, 0:ext_w, 0] = 255

    if int_w > 0:
        frame[1:bar_h - 1, w - int_w:w, 2] = 255

    if success:
        frame[0:2, :, 1] = 255

    return frame


def save_video(video_path, frames, fps):
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(
        str(video_path),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        pixelformat="yuv420p",
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def run_one_episode(
    agent,
    env_id,
    device,
    seed,
    max_steps,
    fps,
    repeat_action_probability,
    deterministic,
    overlay_alpha,
    upscale,
    saliency_every,
):
    envs = make_env(
        env_id=env_id,
        seed=seed,
        repeat_action_probability=repeat_action_probability,
    )

    obs = torch.tensor(envs.reset(), device=device, dtype=torch.float32)

    frames = []
    episode_return = 0.0

    last_ext_sal = None
    last_int_sal = None
    last_v_ext = 0.0
    last_v_int = 0.0

    for step in range(max_steps):
        # Compute saliency every N steps. Reuse last saliency in between to save time.
        if step % saliency_every == 0 or last_ext_sal is None:
            ext_sal, int_sal, v_ext, v_int = compute_value_saliency(agent, obs)
            last_ext_sal = ext_sal
            last_int_sal = int_sal
            last_v_ext = v_ext
            last_v_int = 0.0 if v_int is None else v_int

        frame = make_saliency_overlay_frame(
            obs_tensor=obs,
            ext_saliency=last_ext_sal,
            int_saliency=last_int_sal,
            alpha=overlay_alpha,
            upscale=upscale,
        )

        frame = add_top_bar(
            frame,
            episode_return=episode_return,
            step=step,
            v_ext=last_v_ext,
            v_int=last_v_int,
            success=False,
        )

        frames.append(frame)

        with torch.no_grad():
            action = agent.get_action(obs, deterministic=deterministic)

        next_obs, reward, done, info = envs.step(action.detach().cpu().numpy())

        episode_return += float(reward[0])
        obs = torch.tensor(next_obs, device=device, dtype=torch.float32)

        if bool(done[0]):
            break

    envs.close()

    return {
        "frames": frames,
        "episode_return": episode_return,
        "num_frames": len(frames),
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="MontezumaRevenge-v5")
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--max-steps", type=int, default=27000)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument(
        "--target-return",
        type=float,
        default=399.0,
        help="Keep retrying until an episode reaches this return.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=50,
        help="Safety cap. The script keeps the best episode if target is never reached.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of taking argmax. Useful for trying to reach target return.",
    )
    parser.add_argument(
        "--repeat-action-probability",
        type=float,
        default=0.25,
        help="Sticky action probability. Use 0.25 to match the RND script.",
    )

    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.80,
        help="Overlay strength for red/blue saliency.",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=4,
        help="Nearest-neighbor upscale factor. 4 turns 84x84 into 336x336.",
    )
    parser.add_argument(
        "--saliency-every",
        type=int,
        default=1,
        help="Compute gradients every N env steps. 1 is best quality but slowest.",
    )

    args = parser.parse_args()

    if args.saliency_every < 1:
        raise ValueError("--saliency-every must be >= 1")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_checkpoint(args.checkpoint)
    state_dict = load_checkpoint(ckpt_path, device)
    use_rnd_critic = infer_is_rnd_checkpoint(state_dict)

    print(f"[info] checkpoint: {ckpt_path}")
    print(f"[info] env_id: {args.env_id}")
    print(f"[info] device: {device}")
    print(f"[info] use_rnd_critic: {use_rnd_critic}")

    if not use_rnd_critic:
        print("[warning] This does not look like an RND checkpoint.")
        print("[warning] Blue V_int saliency will be unavailable unless critic_int exists.")

    # Build a temporary env only to get action space and obs space.
    tmp_envs = make_env(
        env_id=args.env_id,
        seed=args.seed,
        repeat_action_probability=args.repeat_action_probability,
    )

    agent = Agent(tmp_envs, use_rnd_critic=use_rnd_critic).to(device)
    tmp_envs.close()

    missing, unexpected = agent.load_state_dict(state_dict, strict=False)

    if missing:
        print("[warning] Missing checkpoint keys:")
        for k in missing:
            print("  ", k)

    if unexpected:
        print("[warning] Unexpected checkpoint keys:")
        for k in unexpected:
            print("  ", k)

    agent.eval()

    best = None
    success = None

    deterministic = not args.stochastic

    print("[info] Starting evaluation attempts")
    print(f"[info] target_return={args.target_return}")
    print(f"[info] max_attempts={args.max_attempts}")
    print(f"[info] policy={'deterministic argmax' if deterministic else 'stochastic sampling'}")

    for attempt in range(1, args.max_attempts + 1):
        attempt_seed = args.seed + attempt - 1

        result = run_one_episode(
            agent=agent,
            env_id=args.env_id,
            device=device,
            seed=attempt_seed,
            max_steps=args.max_steps,
            fps=args.fps,
            repeat_action_probability=args.repeat_action_probability,
            deterministic=deterministic,
            overlay_alpha=args.overlay_alpha,
            upscale=args.upscale,
            saliency_every=args.saliency_every,
        )

        ret = result["episode_return"]
        print(
            f"[attempt {attempt:03d}/{args.max_attempts}] "
            f"seed={attempt_seed} return={ret:.2f} frames={result['num_frames']}"
        )

        if best is None or ret > best["episode_return"]:
            best = result
            print(f"[info] new best return: {ret:.2f}")

        if ret >= args.target_return:
            success = result
            print(f"[success] reached target_return={args.target_return} with return={ret:.2f}")
            break

    kept = success if success is not None else best

    if kept is None or not kept["frames"]:
        raise RuntimeError("No frames captured from any episode.")

    # Mark the kept video as successful with a green top strip if target was reached.
    reached = success is not None
    if reached:
        marked_frames = []
        for f in kept["frames"]:
            f = f.copy()
            f[0:2, :, 1] = 255
            marked_frames.append(f)
        kept["frames"] = marked_frames

    output_path = Path(args.output)
    save_video(output_path, kept["frames"], fps=args.fps)

    print("[done] saved saliency video")
    print(f"[done] output: {output_path}")
    print(f"[done] kept_seed: {kept['seed']}")
    print(f"[done] kept_return: {kept['episode_return']:.2f}")
    print(f"[done] reached_target: {reached}")

    if not reached:
        print(
            "[warning] Target return was not reached. "
            "Saved the best attempt instead. Increase --max-attempts or use --stochastic."
        )


if __name__ == "__main__":
    main()
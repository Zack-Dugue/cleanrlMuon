#!/usr/bin/env python3
import argparse
import glob
import os
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
    Matches the Agent architecture in your attached PPO/RND envpool scripts.

    Plain PPO checkpoint:
      - has network, extra_layer, actor, critic_ext

    RND checkpoint:
      - has network, extra_layer, actor, critic_ext, critic_int

    The actor path is the same, so for video we only need:
      network -> actor
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

    def get_action(self, x, deterministic: bool = True):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)

        if deterministic:
            return torch.argmax(logits, dim=1)

        probs = Categorical(logits=logits)
        return probs.sample()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(path, map_location=device)


def infer_is_rnd_checkpoint(state_dict):
    return any(k.startswith("critic_int.") for k in state_dict.keys())


def make_rgb_frame(obs_tensor):
    """
    EnvPool Atari obs is usually shape:
      [num_envs, 4, 84, 84]

    We take the newest frame from the 4-frame stack:
      obs[0, -1] -> [84, 84]

    Then convert grayscale to RGB for mp4 encoders.
    """
    frame = obs_tensor[0, -1].detach().cpu().numpy().astype(np.uint8)

    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)

    return frame


def save_video_imageio(video_path, frames, fps):
    """
    Uses imageio's writer API instead of mimsave. This tends to be a little
    less weird with ffmpeg kwargs.
    """
    video_path = str(video_path)

    with imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        pixelformat="yuv420p",
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="MontezumaRevenge-v5")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--max-steps", type=int, default=27000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--repeat-action-probability",
        type=float,
        default=0.25,
        help="Use 0.25 to match RND script; set 0.0 if you want non-sticky evaluation.",
    )
    parser.add_argument(
        "--force-rnd-critic",
        action="store_true",
        help="Force construction of the RND Agent with critic_int.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        matches = glob.glob(args.checkpoint)
        if not matches:
            raise FileNotFoundError(f"No checkpoint found matching: {args.checkpoint}")
        ckpt_path = Path(matches[0])
        print(f"[info] Using matched checkpoint: {ckpt_path}")

    state_dict = load_checkpoint(ckpt_path, device)
    use_rnd_critic = args.force_rnd_critic or infer_is_rnd_checkpoint(state_dict)

    print(f"[info] checkpoint: {ckpt_path}")
    print(f"[info] inferred use_rnd_critic={use_rnd_critic}")
    print(f"[info] env_id={args.env_id}")
    print(f"[info] device={device}")

    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=1,
        episodic_life=False,
        reward_clip=False,
        seed=args.seed,
        repeat_action_probability=args.repeat_action_probability,
    )
    envs.num_envs = 1
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    assert isinstance(envs.action_space, gym.spaces.Discrete)

    agent = Agent(envs, use_rnd_critic=use_rnd_critic).to(device)
    missing, unexpected = agent.load_state_dict(state_dict, strict=False)

    if missing:
        print("[warning] Missing keys:")
        for k in missing:
            print("  ", k)

    if unexpected:
        print("[warning] Unexpected keys:")
        for k in unexpected:
            print("  ", k)

    # If the only mismatch is critic_int due to PPO vs RND, that is okay for video.
    agent.eval()

    if args.output is None:
        out_dir = Path("videos_from_checkpoints")
        out_dir.mkdir(exist_ok=True)
        video_path = out_dir / f"{ckpt_path.stem}.{args.env_id}.eval.mp4"
    else:
        video_path = Path(args.output)
        video_path.parent.mkdir(parents=True, exist_ok=True)

    obs = torch.tensor(envs.reset(), device=device, dtype=torch.float32)

    frames = []
    episode_return = 0.0

    for step in range(args.max_steps):
        frames.append(make_rgb_frame(obs))

        with torch.no_grad():
            action = agent.get_action(obs, deterministic=not args.stochastic)

        next_obs, reward, done, info = envs.step(action.detach().cpu().numpy())

        episode_return += float(reward[0])
        obs = torch.tensor(next_obs, device=device, dtype=torch.float32)

        if bool(done[0]):
            print(f"[info] Episode ended at step={step + 1}")
            break

    envs.close()

    if not frames:
        raise RuntimeError("No frames were captured.")

    print(f"[info] Captured {len(frames)} frames")
    print(f"[info] Evaluation return: {episode_return}")
    print(f"[info] Saving video to: {video_path}")

    save_video_imageio(video_path, frames, fps=args.fps)

    print("[done] Saved video successfully.")


if __name__ == "__main__":
    main()
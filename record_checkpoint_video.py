#!/usr/bin/env python3
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, action_dim, use_rnd_critic=False):
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
            layer_init(nn.Linear(448, action_dim), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        if use_rnd_critic:
            self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action(self, x, deterministic=True):
        # x expected shape: [B, 4, 84, 84], uint8/float-like in [0,255]
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
        return torch.load(path, map_location=device)


def infer_is_rnd_checkpoint(state_dict):
    return any(k.startswith("critic_int.") for k in state_dict.keys())


def make_env(env_id, seed):
    # Raw RGB-rendering environment
    env = gym.make(
        f"ALE/{env_id}",
        render_mode="rgb_array",
        frameskip=1,  # avoid double frame-skip because AtariPreprocessing will handle it
    )

    # Match the usual Atari preprocessing as closely as possible
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,   # for eval, full-episode gameplay is usually nicer
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )

    # Stack 4 grayscale frames => shape should become (4, 84, 84)
    env = FrameStackObservation(env, 4)

    obs, info = env.reset(seed=seed)
    return env, obs


def obs_to_tensor(obs, device):
    # FrameStackObservation returns lazy/stacked arrays in HWC-ish or stacked format depending on version.
    # We want [1, 4, 84, 84].
    obs = np.array(obs)

    if obs.shape == (4, 84, 84):
        arr = obs
    elif obs.shape == (84, 84, 4):
        arr = np.transpose(obs, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected obs shape: {obs.shape}")

    return torch.tensor(arr[None, ...], device=device, dtype=torch.float32)


def save_video(video_path, frames, fps=30):
    with imageio.get_writer(
        str(video_path),
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
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-steps", type=int, default=27000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    state_dict = load_checkpoint(ckpt_path, device)
    use_rnd_critic = infer_is_rnd_checkpoint(state_dict)

    # Convert MontezumaRevenge-v5 -> MontezumaRevenge
    env_base = args.env_id.replace("-v5", "")

    env, obs = make_env(env_base, args.seed)
    action_dim = env.action_space.n

    agent = Agent(action_dim=action_dim, use_rnd_critic=use_rnd_critic).to(device)
    missing, unexpected = agent.load_state_dict(state_dict, strict=False)

    if missing:
        print("[warning] Missing keys:")
        for k in missing:
            print(" ", k)

    if unexpected:
        print("[warning] Unexpected keys:")
        for k in unexpected:
            print(" ", k)

    agent.eval()

    frames = []
    episode_return = 0.0

    for step in range(args.max_steps):
        # grab real RGB frame from the environment
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs_tensor = obs_to_tensor(obs, device)

        with torch.no_grad():
            action = agent.get_action(obs_tensor, deterministic=not args.stochastic)

        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        episode_return += float(reward)

        if terminated or truncated:
            print(f"[info] episode ended at step {step + 1}")
            break

    env.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] captured {len(frames)} RGB frames")
    print(f"[info] episode return: {episode_return}")
    print(f"[info] saving to {output_path}")

    save_video(output_path, frames, fps=args.fps)
    print("[done] video saved")


if __name__ == "__main__":
    main()
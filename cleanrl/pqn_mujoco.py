#!/usr/bin/env python3
"""Actor-Critic PQN for continuous-control Gymnasium environments.

This is a PyTorch/CleanRL-style version of the continuous-action PQN idea:

* parallel, short online rollouts;
* Q(lambda) regression without replay or a target network;
* a deterministic actor that approximates argmax_a Q(s, a);
* optional Muon updates on every critic matrix parameter, with Adam on the
  one-dimensional parameters.  The actor stays on Adam by default so that an
  Adam-vs-Muon comparison isolates the Q-network optimizer.

The implementation deliberately exposes a JSON summary contract.  The sibling
tuning and evaluation launchers consume ``summary.json`` directly rather than
scraping console or TensorBoard output.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizers import MuonWithAuxAdam  # noqa: E402


@dataclass
class Args:
    # Experiment plumbing.
    exp_name: str = "pqn_mujoco"
    """Experiment name used in logs and tracking."""
    seed: int = 1
    """Random seed."""
    torch_deterministic: bool = True
    """Use deterministic cuDNN kernels when available."""
    cuda: bool = True
    """Use CUDA when it is available."""
    track: bool = False
    """Track the run with Weights & Biases."""
    wandb_project_name: str = "cleanRL-mujoco-pqn"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    capture_video: bool = False
    """Record evaluation video for the first evaluation episode."""
    output_dir: Optional[str] = None
    """Run directory. Defaults to runs_mujoco/<generated run name>."""
    save_model: bool = False

    # Environment and rollout.
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1_000_000
    num_envs: int = 16
    num_steps: int = 32
    gamma: float = 0.99
    q_lambda: float = 0.95
    start_noise: float = 0.30
    end_noise: float = 0.05
    exploration_fraction: float = 0.60
    obs_norm: bool = True
    obs_norm_clip: float = 10.0

    # Optimization.
    optimizer: str = "Adam"
    """Critic optimizer: Adam or Muon."""
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    num_minibatches: int = 4
    update_epochs: int = 2
    actor_update_frequency: int = 1
    max_grad_norm: float = 10.0
    anneal_lr: bool = False
    huber_delta: float = 10.0
    muon_actor: bool = False
    """Also put all actor matrix parameters on Muon (off by default)."""
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5

    # Model and final evaluation.
    hidden_dim: int = 256
    hidden_layers: int = 2
    norm_type: str = "layernorm"
    """Hidden-layer normalization: layernorm or none."""
    eval_episodes: int = 10
    greedy_eval_steps: int = 0
    """Total environment transitions in an additional fixed-budget greedy evaluation."""
    greedy_eval_num_envs: int = 8
    """Parallel environments used for the fixed-budget greedy evaluation."""


class RunningMeanStd:
    """Numerically stable, mergeable observation moments."""

    def __init__(self, shape: Sequence[int], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x64, axis=0)
        batch_var = np.var(x64, axis=0)
        batch_count = x64.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray, clip: float) -> np.ndarray:
        normalized = (np.asarray(x, dtype=np.float32) - self.mean) / np.sqrt(
            self.var + 1e-8
        )
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def state_dict(self) -> Dict[str, object]:
        return {"mean": self.mean, "var": self.var, "count": self.count}


def layer_init(
    layer: nn.Linear, std: float = math.sqrt(2.0), bias_const: float = 0.0
) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def make_norm(norm_type: str, dim: int) -> nn.Module:
    normalized = norm_type.strip().lower()
    if normalized in {"layernorm", "layer_norm", "ln"}:
        return nn.LayerNorm(dim)
    if normalized in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(
        f"Unknown norm_type={norm_type!r}; expected 'layernorm' or 'none'."
    )


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int,
        hidden_layers: int,
        norm_type: str,
    ):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least one")
        blocks: List[nn.Module] = []
        in_dim = obs_dim
        for _ in range(hidden_layers):
            blocks.extend(
                [
                    layer_init(nn.Linear(in_dim, hidden_dim)),
                    make_norm(norm_type, hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*blocks)
        self.output = layer_init(nn.Linear(hidden_dim, action_low.size), std=0.01)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer(
            "action_scale", torch.as_tensor(action_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.as_tensor(action_bias, dtype=torch.float32)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (
            torch.tanh(self.output(self.trunk(obs))) * self.action_scale
            + self.action_bias
        )


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        norm_type: str,
    ):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least one")
        blocks: List[nn.Module] = []
        in_dim = obs_dim + action_dim
        for _ in range(hidden_layers):
            blocks.extend(
                [
                    layer_init(nn.Linear(in_dim, hidden_dim)),
                    make_norm(norm_type, hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*blocks)
        self.output = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.output(self.trunk(torch.cat((obs, action), dim=-1))).squeeze(-1)


def split_matrix_parameters(
    module: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Put every matrix-shaped trainable parameter on Muon, including output matrices."""

    matrix_params: List[nn.Parameter] = []
    aux_params: List[nn.Parameter] = []
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            matrix_params.append(parameter)
        else:
            aux_params.append(parameter)
    return matrix_params, aux_params


def make_optimizer(
    module: nn.Module,
    name: str,
    learning_rate: float,
    *,
    muon_momentum: float,
    muon_nesterov: bool,
    muon_ns_steps: int,
) -> torch.optim.Optimizer:
    normalized = name.strip().lower()
    if normalized == "adam":
        return torch.optim.Adam(
            module.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8
        )
    if normalized != "muon":
        raise ValueError(f"Unknown optimizer={name!r}; expected Adam or Muon.")

    matrix_params, aux_params = split_matrix_parameters(module)
    groups: List[Dict[str, object]] = [
        {
            "params": matrix_params,
            "use_muon": True,
            "lr": learning_rate,
            "momentum": muon_momentum,
            "nesterov": muon_nesterov,
            "ns_steps": muon_ns_steps,
            "weight_decay": 0.0,
        }
    ]
    if aux_params:
        groups.append(
            {
                "params": aux_params,
                "use_muon": False,
                "lr": learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
        )
    return MuonWithAuxAdam(groups)


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def linear_schedule(start: float, end: float, duration: float, t: int) -> float:
    if duration <= 0:
        return end
    fraction = min(max(t / duration, 0.0), 1.0)
    return start + fraction * (end - start)


def compute_q_lambda_targets(
    rewards: torch.Tensor,
    episode_ends: torch.Tensor,
    terminated: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    q_lambda: float,
) -> torch.Tensor:
    """Compute Q(lambda) targets, correctly bootstrapping time-limit truncations.

    ``episode_ends`` cuts the lambda trace on either termination or truncation.
    ``terminated`` suppresses bootstrapping only for true MDP terminal states.
    """

    returns = torch.empty_like(rewards)
    final_index = rewards.shape[0] - 1
    for t in reversed(range(rewards.shape[0])):
        one_step = rewards[t] + gamma * (1.0 - terminated[t]) * next_values[t]
        if t == final_index:
            returns[t] = one_step
        else:
            continued = rewards[t] + gamma * (
                (1.0 - q_lambda) * next_values[t] + q_lambda * returns[t + 1]
            )
            returns[t] = torch.where(episode_ends[t] > 0.0, one_step, continued)
    return returns


def final_observations(next_obs: np.ndarray, infos: Dict[str, object]) -> np.ndarray:
    """Recover the actual final state hidden by Gymnasium vector autoreset."""

    bootstrap_obs = np.asarray(next_obs, dtype=np.float32).copy()
    observations = infos.get("final_observation")
    if observations is None:
        return bootstrap_obs
    mask = infos.get("_final_observation")
    for idx, observation in enumerate(observations):
        if observation is not None and (mask is None or bool(mask[idx])):
            bootstrap_obs[idx] = np.asarray(observation, dtype=np.float32)
    return bootstrap_obs


def make_env(env_id: str, seed: int, idx: int):
    def thunk():
        env = gym.make(env_id)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def evaluate_actor(
    actor: Actor,
    env_id: str,
    seed: int,
    episodes: int,
    obs_rms: RunningMeanStd,
    obs_norm: bool,
    obs_norm_clip: float,
    device: torch.device,
    video_dir: Optional[Path] = None,
) -> Tuple[List[float], List[int]]:
    returns: List[float] = []
    lengths: List[int] = []
    for episode in range(episodes):
        render_mode = "rgb_array" if video_dir is not None and episode == 0 else None
        env = gym.make(env_id, render_mode=render_mode)
        if video_dir is not None and episode == 0:
            env = gym.wrappers.RecordVideo(
                env, str(video_dir), episode_trigger=lambda _: True
            )
        obs, _ = env.reset(seed=seed + 100_000 + episode)
        episode_return = 0.0
        episode_length = 0
        done = False
        while not done:
            obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            if obs_norm:
                obs_batch = obs_rms.normalize(obs_batch, obs_norm_clip)
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                action = actor(obs_tensor).cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            done = bool(terminated or truncated)
        env.close()
        returns.append(episode_return)
        lengths.append(episode_length)
    return returns, lengths


def evaluate_actor_for_steps(
    actor: Actor,
    env_id: str,
    seed: int,
    total_steps: int,
    num_envs: int,
    obs_rms: RunningMeanStd,
    obs_norm: bool,
    obs_norm_clip: float,
    device: torch.device,
) -> Tuple[List[float], List[int], int, float]:
    """Evaluate the noise-free greedy actor for a fixed interaction budget.

    Only completed episodes contribute to the return statistic. Because vector
    environments step together, the actual transition count is rounded up to a
    multiple of ``num_envs``.
    """

    if total_steps <= 0:
        return [], [], 0, 0.0
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed + 200_000, index) for index in range(num_envs)]
    )
    obs, _ = eval_envs.reset(seed=seed + 200_000)
    obs = np.asarray(obs, dtype=np.float32).reshape(num_envs, -1)
    running_returns = np.zeros(num_envs, dtype=np.float64)
    running_lengths = np.zeros(num_envs, dtype=np.int64)
    completed_returns: List[float] = []
    completed_lengths: List[int] = []
    vector_steps = math.ceil(total_steps / num_envs)
    evaluation_start = time.time()

    actor.eval()
    for _ in range(vector_steps):
        normalized_obs = obs_rms.normalize(obs, obs_norm_clip) if obs_norm else obs
        obs_tensor = torch.as_tensor(normalized_obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            # In continuous action spaces the actor is the learned approximation
            # to argmax_a Q(s, a). No exploration noise is added here.
            action = actor(obs_tensor).cpu().numpy()
        obs, rewards, terminated, truncated, _ = eval_envs.step(action)
        obs = np.asarray(obs, dtype=np.float32).reshape(num_envs, -1)
        running_returns += np.asarray(rewards, dtype=np.float64)
        running_lengths += 1
        ended = np.logical_or(terminated, truncated)
        for env_index in np.flatnonzero(ended):
            completed_returns.append(float(running_returns[env_index]))
            completed_lengths.append(int(running_lengths[env_index]))
            running_returns[env_index] = 0.0
            running_lengths[env_index] = 0

    eval_envs.close()
    elapsed = time.time() - evaluation_start
    actual_steps = vector_steps * num_envs
    if not completed_returns:
        raise RuntimeError(
            f"greedy_eval_steps={total_steps} completed no episodes; increase the budget "
            f"for {env_id}"
        )
    return completed_returns, completed_lengths, actual_steps, elapsed


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def mean_or_nan(values: Iterable[float]) -> float:
    values_list = list(values)
    return float(np.mean(values_list)) if values_list else float("nan")


def validate_args(args: Args) -> None:
    if args.num_envs <= 0 or args.num_steps <= 0:
        raise ValueError("num_envs and num_steps must be positive")
    batch_size = args.num_envs * args.num_steps
    if args.num_minibatches <= 0 or batch_size % args.num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    if args.total_timesteps < batch_size:
        raise ValueError("total_timesteps must contain at least one rollout batch")
    if not 0.0 <= args.q_lambda <= 1.0:
        raise ValueError("q_lambda must be in [0, 1]")
    if args.actor_update_frequency <= 0:
        raise ValueError("actor_update_frequency must be positive")
    if args.eval_episodes < 0 or args.greedy_eval_steps < 0:
        raise ValueError("eval_episodes and greedy_eval_steps must be non-negative")
    if args.greedy_eval_num_envs <= 0:
        raise ValueError("greedy_eval_num_envs must be positive")


def main() -> None:
    args = tyro.cli(Args)
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.optimizer}__seed{args.seed}__{int(time.time())}"
    run_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "runs_mujoco" / run_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", asdict(args))

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            sync_tensorboard=True,
            config=asdict(args),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(str(run_dir / "tb"))
    writer.add_text(
        "hyperparameters",
        "|parameter|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in asdict(args).items()),
    )

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, idx) for idx in range(args.num_envs)]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("pqn_mujoco.py requires a continuous Box action space")
    if len(envs.single_observation_space.shape) != 1:
        raise TypeError("pqn_mujoco.py currently requires flat vector observations")
    if not np.all(np.isfinite(envs.single_action_space.low)) or not np.all(
        np.isfinite(envs.single_action_space.high)
    ):
        raise ValueError("The action space must have finite lower and upper bounds")

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    actor = Actor(
        obs_dim,
        envs.single_action_space.low.reshape(-1),
        envs.single_action_space.high.reshape(-1),
        args.hidden_dim,
        args.hidden_layers,
        args.norm_type,
    ).to(device)
    critic = Critic(
        obs_dim, action_dim, args.hidden_dim, args.hidden_layers, args.norm_type
    ).to(device)

    actor_optimizer_name = "Muon" if args.muon_actor else "Adam"
    actor_optimizer = make_optimizer(
        actor,
        actor_optimizer_name,
        args.actor_learning_rate,
        muon_momentum=args.muon_momentum,
        muon_nesterov=args.muon_nesterov,
        muon_ns_steps=args.muon_ns_steps,
    )
    critic_optimizer = make_optimizer(
        critic,
        args.optimizer,
        args.critic_learning_rate,
        muon_momentum=args.muon_momentum,
        muon_nesterov=args.muon_nesterov,
        muon_ns_steps=args.muon_ns_steps,
    )

    obs_buffer = torch.zeros(
        (args.num_steps, args.num_envs, obs_dim), dtype=torch.float32, device=device
    )
    action_buffer = torch.zeros(
        (args.num_steps, args.num_envs, action_dim), dtype=torch.float32, device=device
    )
    reward_buffer = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.float32, device=device
    )
    end_buffer = torch.zeros_like(reward_buffer)
    terminated_buffer = torch.zeros_like(reward_buffer)
    next_value_buffer = torch.zeros_like(reward_buffer)

    obs_rms = RunningMeanStd((obs_dim,))
    next_obs_raw, _ = envs.reset(seed=args.seed)
    next_obs_raw = np.asarray(next_obs_raw, dtype=np.float32).reshape(
        args.num_envs, obs_dim
    )
    obs_rms.update(next_obs_raw)
    episode_returns_running = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths_running = np.zeros(args.num_envs, dtype=np.int64)
    completed_returns: List[float] = []
    completed_lengths: List[int] = []
    progress_rows: List[Dict[str, object]] = []
    global_step = 0
    critic_gradient_steps = 0
    start_time = time.time()
    last_critic_loss = float("nan")
    last_actor_loss = float("nan")
    last_q_mean = float("nan")

    print(
        f"{args.optimizer} critic / {actor_optimizer_name} actor | env={args.env_id} | "
        f"seed={args.seed} | device={device} | batch={batch_size}"
    )

    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / max(num_iterations, 1)
            set_learning_rate(critic_optimizer, args.critic_learning_rate * fraction)
            set_learning_rate(actor_optimizer, args.actor_learning_rate * fraction)

        exploration_noise = linear_schedule(
            args.start_noise,
            args.end_noise,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        for step in range(args.num_steps):
            normalized_obs = (
                obs_rms.normalize(next_obs_raw, args.obs_norm_clip)
                if args.obs_norm
                else next_obs_raw
            )
            obs_tensor = torch.as_tensor(
                normalized_obs, dtype=torch.float32, device=device
            )
            obs_buffer[step].copy_(obs_tensor)

            with torch.no_grad():
                deterministic_action = actor(obs_tensor)
                noise = (
                    torch.randn_like(deterministic_action)
                    * exploration_noise
                    * actor.action_scale
                )
                behavior_action = deterministic_action + noise
                behavior_action = torch.maximum(
                    torch.minimum(
                        behavior_action, actor.action_bias + actor.action_scale
                    ),
                    actor.action_bias - actor.action_scale,
                )
            action_buffer[step].copy_(behavior_action)

            stepped_obs, reward, terminated, truncated, infos = envs.step(
                behavior_action.cpu().numpy()
            )
            stepped_obs = np.asarray(stepped_obs, dtype=np.float32).reshape(
                args.num_envs, obs_dim
            )
            reward_array = np.asarray(reward, dtype=np.float32)
            terminated_array = np.asarray(terminated, dtype=np.float32)
            ended_array = np.logical_or(terminated, truncated)
            bootstrap_obs_raw = final_observations(stepped_obs, infos).reshape(
                args.num_envs, obs_dim
            )

            # Include final states hidden by autoreset without double-counting every
            # ordinary next state.
            stats_observations = stepped_obs
            final_observation_values = infos.get("final_observation")
            if final_observation_values is not None:
                final_mask = infos.get("_final_observation")
                extra_final_observations = [
                    np.asarray(observation, dtype=np.float32).reshape(obs_dim)
                    for index, observation in enumerate(final_observation_values)
                    if observation is not None
                    and (final_mask is None or bool(final_mask[index]))
                ]
                if extra_final_observations:
                    stats_observations = np.concatenate(
                        (stepped_obs, np.stack(extra_final_observations)), axis=0
                    )
            obs_rms.update(stats_observations)
            normalized_bootstrap = (
                obs_rms.normalize(bootstrap_obs_raw, args.obs_norm_clip)
                if args.obs_norm
                else bootstrap_obs_raw
            )
            bootstrap_tensor = torch.as_tensor(
                normalized_bootstrap, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                greedy_next_action = actor(bootstrap_tensor)
                next_value_buffer[step] = critic(bootstrap_tensor, greedy_next_action)

            reward_buffer[step] = torch.as_tensor(reward_array, device=device)
            end_buffer[step] = torch.as_tensor(
                ended_array, dtype=torch.float32, device=device
            )
            terminated_buffer[step] = torch.as_tensor(terminated_array, device=device)

            episode_returns_running += reward_array
            episode_lengths_running += 1
            for env_index in np.flatnonzero(ended_array):
                episode_return = float(episode_returns_running[env_index])
                episode_length = int(episode_lengths_running[env_index])
                completed_returns.append(episode_return)
                completed_lengths.append(episode_length)
                writer.add_scalar(
                    "charts/episodic_return",
                    episode_return,
                    global_step + args.num_envs,
                )
                writer.add_scalar(
                    "charts/episodic_length",
                    episode_length,
                    global_step + args.num_envs,
                )
                episode_returns_running[env_index] = 0.0
                episode_lengths_running[env_index] = 0

            next_obs_raw = stepped_obs
            global_step += args.num_envs

        with torch.no_grad():
            target_buffer = compute_q_lambda_targets(
                reward_buffer,
                end_buffer,
                terminated_buffer,
                next_value_buffer,
                args.gamma,
                args.q_lambda,
            )

        batch_obs = obs_buffer.reshape(batch_size, obs_dim)
        batch_actions = action_buffer.reshape(batch_size, action_dim)
        batch_targets = target_buffer.reshape(batch_size)
        indices = np.arange(batch_size)
        critic_losses: List[float] = []
        actor_losses: List[float] = []
        q_means: List[float] = []

        for _ in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                minibatch_indices = torch.as_tensor(
                    indices[start : start + minibatch_size],
                    dtype=torch.long,
                    device=device,
                )
                mb_obs = batch_obs[minibatch_indices]
                mb_actions = batch_actions[minibatch_indices]
                mb_targets = batch_targets[minibatch_indices]

                predicted_q = critic(mb_obs, mb_actions)
                critic_loss = F.huber_loss(
                    predicted_q, mb_targets, delta=args.huber_delta
                )
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                critic_optimizer.step()
                critic_gradient_steps += 1
                critic_losses.append(float(critic_loss.detach()))
                q_means.append(float(predicted_q.detach().mean()))

                if critic_gradient_steps % args.actor_update_frequency == 0:
                    # Freeze critic parameters but retain dQ/da for the actor.
                    critic.requires_grad_(False)
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_action = actor(mb_obs)
                    actor_loss = -critic(mb_obs, actor_action).mean()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                    actor_optimizer.step()
                    critic.requires_grad_(True)
                    actor_losses.append(float(actor_loss.detach()))

        last_critic_loss = mean_or_nan(critic_losses)
        last_actor_loss = mean_or_nan(actor_losses)
        last_q_mean = mean_or_nan(q_means)
        sps = int(global_step / max(time.time() - start_time, 1e-9))
        writer.add_scalar("losses/critic_loss", last_critic_loss, global_step)
        writer.add_scalar("losses/actor_loss", last_actor_loss, global_step)
        writer.add_scalar("losses/q_values", last_q_mean, global_step)
        writer.add_scalar("charts/exploration_noise", exploration_noise, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        log_every = max(1, num_iterations // 20)
        if iteration == 1 or iteration % log_every == 0 or iteration == num_iterations:
            recent_return = mean_or_nan(completed_returns[-20:])
            row = {
                "iteration": iteration,
                "global_step": global_step,
                "critic_loss": last_critic_loss,
                "actor_loss": last_actor_loss,
                "q_mean": last_q_mean,
                "exploration_noise": exploration_noise,
                "return_last_20": recent_return,
                "sps": sps,
            }
            progress_rows.append(row)
            print(
                f"step={global_step:,} critic_loss={last_critic_loss:.5f} "
                f"actor_loss={last_actor_loss:.5f} return20={recent_return:.2f} SPS={sps}"
            )

    training_seconds = time.time() - start_time
    final_sps = int(global_step / max(training_seconds, 1e-9))
    video_dir = run_dir / "videos" if args.capture_video else None
    eval_returns, eval_lengths = evaluate_actor(
        actor,
        args.env_id,
        args.seed,
        args.eval_episodes,
        obs_rms,
        args.obs_norm,
        args.obs_norm_clip,
        device,
        video_dir,
    )
    (
        greedy_eval_returns,
        greedy_eval_lengths,
        greedy_eval_steps_actual,
        greedy_eval_seconds,
    ) = evaluate_actor_for_steps(
        actor,
        args.env_id,
        args.seed,
        args.greedy_eval_steps,
        args.greedy_eval_num_envs,
        obs_rms,
        args.obs_norm,
        args.obs_norm_clip,
        device,
    )
    summary: Dict[str, object] = {
        "status": "ok",
        "algorithm": "actor_critic_pqn",
        "env_id": args.env_id,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "actor_optimizer": actor_optimizer_name,
        "global_step": global_step,
        "requested_total_timesteps": args.total_timesteps,
        "training_seconds": training_seconds,
        "sps": final_sps,
        "eval_episodes": args.eval_episodes,
        "eval_returns": eval_returns,
        "eval_lengths": eval_lengths,
        "eval_mean_return": mean_or_nan(eval_returns),
        "eval_median_return": float(np.median(eval_returns))
        if eval_returns
        else float("nan"),
        "eval_std_return": float(np.std(eval_returns))
        if eval_returns
        else float("nan"),
        "greedy_eval_steps_requested": args.greedy_eval_steps,
        "greedy_eval_steps_actual": greedy_eval_steps_actual,
        "greedy_eval_num_envs": args.greedy_eval_num_envs,
        "greedy_eval_seconds": greedy_eval_seconds,
        "greedy_eval_episodes": len(greedy_eval_returns),
        "greedy_eval_returns": greedy_eval_returns,
        "greedy_eval_lengths": greedy_eval_lengths,
        "greedy_eval_mean_return": mean_or_nan(greedy_eval_returns),
        "greedy_eval_median_return": float(np.median(greedy_eval_returns))
        if greedy_eval_returns
        else float("nan"),
        "greedy_eval_std_return": float(np.std(greedy_eval_returns))
        if greedy_eval_returns
        else float("nan"),
        "training_episodes": len(completed_returns),
        "training_return_mean_last_20": mean_or_nan(completed_returns[-20:]),
        "training_return_mean": mean_or_nan(completed_returns),
        "critic_loss": last_critic_loss,
        "actor_loss": last_actor_loss,
        "q_mean": last_q_mean,
        "config": asdict(args),
    }
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)

    if progress_rows:
        with (run_dir / "progress.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=list(progress_rows[0]))
            writer_csv.writeheader()
            writer_csv.writerows(progress_rows)
    if args.save_model:
        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "obs_rms": obs_rms.state_dict(),
                "config": asdict(args),
            },
            run_dir / "model.pt",
        )

    if eval_returns:
        writer.add_scalar("eval/mean_return", summary["eval_mean_return"], global_step)
    if greedy_eval_returns:
        writer.add_scalar(
            "eval/greedy_fixed_budget_mean_return",
            summary["greedy_eval_mean_return"],
            global_step,
        )
    writer.close()
    envs.close()
    if args.track:
        import wandb

        evaluation_metrics = {}
        if eval_returns:
            evaluation_metrics["eval/mean_return"] = summary["eval_mean_return"]
        if greedy_eval_returns:
            evaluation_metrics["eval/greedy_fixed_budget_mean_return"] = summary[
                "greedy_eval_mean_return"
            ]
        wandb.log(evaluation_metrics, step=global_step)
        wandb.finish()
    if eval_returns:
        print(f"eval_mean_return={summary['eval_mean_return']:.3f}")
    if greedy_eval_returns:
        print(
            f"greedy_eval_mean_return={summary['greedy_eval_mean_return']:.3f} "
            f"episodes={len(greedy_eval_returns)} steps={greedy_eval_steps_actual}"
        )
    print(f"RESULT_JSON={summary_path.resolve()}")


if __name__ == "__main__":
    main()

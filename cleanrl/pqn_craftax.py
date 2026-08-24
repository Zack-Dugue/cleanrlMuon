#!/usr/bin/env python3
"""PyTorch PQN on Craftax symbolic observations through a JAX environment bridge.

The learner intentionally follows the repository's replay-free, target-network-free
PQN structure. Craftax owns only environment state and stepping; observations,
actions, and rewards cross the JAX/PyTorch boundary with DLPack.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# These must be set before JAX is imported. PyTorch also needs room on the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizers import MuonWithAuxAdam  # noqa: E402


@dataclass
class Args:
    exp_name: str = "pqn_craftax"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    device: Optional[str] = None

    track: bool = True
    wandb_project_name: str = "cleanRL-craftax-pqn"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tag: Optional[str] = None
    output_dir: Optional[str] = None
    save_model: bool = False

    # Classic is the closest Craftax counterpart to Crafter symbolic. The full
    # environment works with the same code via --env-id Craftax-Symbolic-v1.
    env_id: str = "Craftax-Classic-Symbolic-v1"
    total_timesteps: int = 1_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 128
    num_steps: int = 64
    num_minibatches: int = 4
    update_epochs: int = 4
    anneal_lr: bool = True
    gamma: float = 0.99
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    max_grad_norm: float = 10.0
    q_lambda: float = 0.90

    hidden_dim: int = 256
    hidden_layers: int = 2
    norm_type: str = "layernorm"

    optimizer: str = "Adam"
    momentum: float = 0.95
    weight_decay: float = 1.0e-4
    use_muon_input: bool = True
    use_muon_output: bool = False
    muon_ns_steps: int = 0

    eval_steps: int = 100_000
    eval_num_envs: int = 32
    log_interval: int = 1
    jax_mem_fraction: float = 0.35

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def linear_schedule(start: float, end: float, duration: float, step: int) -> float:
    if duration <= 0:
        return end
    fraction = min(max(float(step) / float(duration), 0.0), 1.0)
    return start + fraction * (end - start)


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class CraftaxTorchBatch:
    """A jitted, vectorized Craftax environment with Torch-facing tensors."""

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        torch_device: torch.device,
    ) -> None:
        if torch_device.type == "cpu":
            # Keep both sides of DLPack on CPU for explicit CPU runs.
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        try:
            import jax
            import jax.numpy as jnp
            from craftax.craftax_env import make_craftax_env_from_name
        except ImportError as error:
            raise RuntimeError(
                "Craftax/JAX is missing. Install Craftax and a JAX build matching "
                "your CUDA version; see craftax_pqn/README.md."
            ) from error

        self.jax = jax
        self.jnp = jnp
        self.env_id = env_id
        self.num_envs = int(num_envs)
        self.torch_device = torch_device
        if torch_device.type == "cuda":
            try:
                gpu_devices = jax.devices("gpu")
            except RuntimeError as error:
                raise RuntimeError(
                    "PyTorch is using CUDA but JAX could not initialize a GPU "
                    "backend. Install the CUDA-enabled JAX wheel."
                ) from error
            if not gpu_devices:
                raise RuntimeError(
                    "PyTorch is using CUDA but JAX has no GPU device. Install the "
                    "CUDA-enabled JAX wheel before running Craftax."
                )
            torch_index = (
                torch_device.index
                if torch_device.index is not None
                else torch.cuda.current_device()
            )
            if torch_index >= len(gpu_devices):
                raise RuntimeError(
                    f"PyTorch selected cuda:{torch_index}, but JAX exposes only "
                    f"{len(gpu_devices)} GPU device(s). Use CUDA_VISIBLE_DEVICES to "
                    "give both frameworks the same single GPU."
                )
            self.jax_device = gpu_devices[torch_index]
        else:
            self.jax_device = jax.devices("cpu")[0]
        self.env = make_craftax_env_from_name(env_id, auto_reset=True)
        self.env_params = self.env.default_params
        observation_space = self.env.observation_space(self.env_params)
        action_space = self.env.action_space(self.env_params)
        self.observation_shape = tuple(int(value) for value in observation_space.shape)
        self.observation_dim = int(np.prod(self.observation_shape))
        self.action_dim = int(action_space.n)

        # Closing over the immutable environment parameters makes the compiled
        # call signatures exactly (keys, states[, actions]).
        self._reset_batch = jax.jit(
            jax.vmap(lambda key: self.env.reset(key, self.env_params))
        )
        self._step_batch = jax.jit(
            jax.vmap(
                lambda key, state, action: self.env.step(
                    key, state, action, self.env_params
                )
            )
        )
        self._key = jax.device_put(jax.random.PRNGKey(int(seed)), self.jax_device)
        self._state: Any = None
        self.episode_returns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=torch_device
        )
        self.episode_lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=torch_device
        )

    def _keys(self) -> Any:
        self._key, batch_key = self.jax.random.split(self._key)
        return self.jax.random.split(batch_key, self.num_envs)

    def _jax_to_torch(self, value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # Both frameworks implement the modern Python DLPack protocol. No host
        # copy is made when JAX and Torch are on the same GPU.
        tensor = torch.utils.dlpack.from_dlpack(value)
        if tensor.device != self.torch_device:
            tensor = tensor.to(self.torch_device)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _torch_to_jax(self, value: torch.Tensor) -> Any:
        value = value.detach().contiguous()
        return self.jax.dlpack.from_dlpack(value)

    def reset(self) -> torch.Tensor:
        observations, self._state = self._reset_batch(self._keys())
        observations = self.jax.block_until_ready(observations)
        self.episode_returns.zero_()
        self.episode_lengths.zero_()
        return self._jax_to_torch(observations, torch.float32).reshape(
            self.num_envs, self.observation_dim
        )

    @torch.no_grad()
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if actions.numel() != self.num_envs:
            raise ValueError("one action is required per Craftax environment")
        jax_actions = self._torch_to_jax(
            actions.reshape(self.num_envs).to(dtype=torch.int32)
        )
        observations, self._state, rewards, dones, _ = self._step_batch(
            self._keys(), self._state, jax_actions
        )
        observations = self.jax.block_until_ready(observations)
        observations_t = self._jax_to_torch(observations, torch.float32).reshape(
            self.num_envs, self.observation_dim
        )
        rewards_t = self._jax_to_torch(rewards, torch.float32).reshape(-1)
        dones_t = self._jax_to_torch(dones, torch.bool).reshape(-1)

        self.episode_returns.add_(rewards_t)
        self.episode_lengths.add_(1)
        completed_returns = self.episode_returns[dones_t].clone()
        completed_lengths = self.episode_lengths[dones_t].clone()
        if torch.any(dones_t):
            self.episode_returns[dones_t] = 0.0
            self.episode_lengths[dones_t] = 0
        return observations_t, rewards_t, dones_t.float(), {
            "episode_returns": completed_returns,
            "episode_lengths": completed_lengths,
        }


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        norm_type: str,
        use_muon_input: bool,
        use_muon_output: bool,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        normalized = norm_type.lower().replace("_", "")
        if normalized not in {"layernorm", "none"}:
            raise ValueError("norm_type must be 'layernorm' or 'none'")
        self.use_muon_input = bool(use_muon_input)
        self.use_muon_output = bool(use_muon_output)
        self.input_layer = layer_init(nn.Linear(observation_dim, hidden_dim))
        self.hidden = nn.ModuleList(
            [layer_init(nn.Linear(hidden_dim, hidden_dim)) for _ in range(hidden_layers - 1)]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(hidden_dim) if normalized == "layernorm" else nn.Identity()
                for _ in range(hidden_layers)
            ]
        )
        self.q_head = layer_init(nn.Linear(hidden_dim, action_dim), std=1.0)

    def features(self, observations: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norms[0](self.input_layer(observations.float())))
        for layer, norm in zip(self.hidden, self.norms[1:]):
            x = F.relu(norm(layer(x)))
        return x

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.features(observations))

    def muon_and_aux_parameters(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        selected_ids = set()
        muon: List[nn.Parameter] = []

        def select(parameter: nn.Parameter) -> None:
            if id(parameter) not in selected_ids:
                selected_ids.add(id(parameter))
                muon.append(parameter)

        if self.use_muon_input:
            select(self.input_layer.weight)
        for layer in self.hidden:
            select(layer.weight)
        if self.use_muon_output:
            select(self.q_head.weight)
        auxiliary = [parameter for parameter in self.parameters() if id(parameter) not in selected_ids]
        return muon, auxiliary


def make_optimizer(q_network: QNetwork, args: Args, device: torch.device):
    name = args.optimizer.lower()
    if name == "adam":
        return optim.Adam(
            q_network.parameters(),
            lr=args.learning_rate,
            betas=(args.momentum, 0.99),
            eps=1.0e-5,
        )
    if name != "muon":
        raise ValueError("optimizer must be Adam or Muon")
    muon_params, auxiliary_params = q_network.muon_and_aux_parameters()
    if not muon_params:
        raise ValueError("Muon was selected but no matrix parameters were routed to Muon")
    ns_steps = args.muon_ns_steps or (5 if device.type == "cuda" else 2)
    return MuonWithAuxAdam(
        [
            {
                "params": muon_params,
                "use_muon": True,
                "lr": args.learning_rate,
                "momentum": args.momentum,
                "nesterov": True,
                "ns_steps": ns_steps,
                "weight_decay": args.weight_decay,
            },
            {
                "params": auxiliary_params,
                "use_muon": False,
                "lr": args.learning_rate,
                "betas": (args.momentum, 0.99),
                "eps": 1.0e-5,
                "weight_decay": args.weight_decay,
            },
        ]
    )


def set_learning_rate(optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def log_metrics(
    writer: SummaryWriter, wandb_run, metrics: Dict[str, float], global_step: int
) -> None:
    for key, value in metrics.items():
        writer.add_scalar(key, value, global_step)
    if wandb_run is not None:
        wandb_run.log(metrics, step=global_step)


@torch.no_grad()
def greedy_evaluation(
    q_network: QNetwork,
    env_id: str,
    num_envs: int,
    total_steps: int,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    q_network.eval()
    env = CraftaxTorchBatch(env_id, num_envs, seed, device)
    observations = env.reset()
    returns: List[torch.Tensor] = []
    lengths: List[torch.Tensor] = []
    reward_sum = torch.zeros(num_envs, dtype=torch.float32, device=device)
    vector_steps = max(1, int(np.ceil(total_steps / num_envs)))
    for _ in range(vector_steps):
        actions = q_network(observations).argmax(dim=-1)
        observations, rewards, _, info = env.step(actions)
        reward_sum.add_(rewards)
        if info["episode_returns"].numel():
            returns.append(info["episode_returns"])
            lengths.append(info["episode_lengths"])

    if returns:
        episode_returns = torch.cat(returns)
        episode_lengths = torch.cat(lengths).float()
        mean_return = float(episode_returns.mean().item())
        std_return = float(episode_returns.std(unbiased=False).item())
        mean_length = float(episode_lengths.mean().item())
        episodes = int(episode_returns.numel())
        fallback = 0.0
    else:
        # This is only a safety fallback for unusually long horizons. It is
        # flagged so it cannot silently masquerade as completed-episode return.
        mean_return = float(reward_sum.mean().item())
        std_return = float(reward_sum.std(unbiased=False).item())
        mean_length = float(vector_steps)
        episodes = 0
        fallback = 1.0
    q_network.train()
    return {
        "eval_greedy_return": mean_return,
        "eval_greedy_return_std": std_return,
        "eval_mean_episode_length": mean_length,
        "eval_episodes": episodes,
        "eval_partial_return_fallback": fallback,
        "eval_reward_per_1000_steps": float(
            reward_sum.sum().item() / max(vector_steps * num_envs, 1) * 1000.0
        ),
        "eval_actual_steps": int(vector_steps * num_envs),
    }


def validate_args(args: Args) -> None:
    if args.num_envs < 1 or args.num_steps < 1:
        raise ValueError("num_envs and num_steps must be positive")
    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size % args.num_minibatches:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.num_iterations < 1:
        raise ValueError("total_timesteps must contain at least one complete rollout")
    if not 0.0 <= args.q_lambda <= 1.0:
        raise ValueError("q_lambda must be in [0, 1]")
    if args.update_epochs < 1:
        raise ValueError("update_epochs must be positive")
    if args.eval_steps < 1 or args.eval_num_envs < 1:
        raise ValueError("eval_steps and eval_num_envs must be positive")
    if not 0.0 < args.jax_mem_fraction <= 1.0:
        raise ValueError("jax_mem_fraction must be in (0, 1]")


def main() -> None:
    args = tyro.cli(Args)
    validate_args(args)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.jax_mem_fraction)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Build and compile the JAX environment before the training clock starts.
    env = CraftaxTorchBatch(args.env_id, args.num_envs, args.seed, device)
    next_observation = env.reset()
    run_name = (
        f"{args.env_id}__{args.exp_name}__{args.optimizer}__"
        f"seed{args.seed}__{int(time.time())}"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else REPO_ROOT / "logs" / "craftax" / "runs" / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = asdict(args)
    config.update(
        {
            "resolved_device": str(device),
            "observation_dim": env.observation_dim,
            "action_dim": env.action_dim,
        }
    )
    write_json(output_dir / "config.json", config)

    q_network = QNetwork(
        env.observation_dim,
        env.action_dim,
        args.hidden_dim,
        args.hidden_layers,
        args.norm_type,
        args.use_muon_input,
        args.use_muon_output,
    ).to(device)
    optimizer = make_optimizer(q_network, args, device)

    wandb_run = None
    if args.track:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            tags=[args.wandb_tag] if args.wandb_tag else None,
            config=config,
            name=run_name,
            dir=str(output_dir),
            save_code=True,
        )

    writer = SummaryWriter(str(output_dir / "tensorboard"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in config.items()),
    )
    progress_path = output_dir / "progress.csv"
    progress_fields = [
        "global_step",
        "iteration",
        "learning_rate",
        "epsilon",
        "episodic_return_mean",
        "episodic_length_mean",
        "td_loss",
        "q_value_mean",
        "grad_norm",
        "sps",
    ]

    observations = torch.zeros(
        (args.num_steps, args.num_envs, env.observation_dim),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.long, device=device
    )
    rewards = torch.zeros_like(actions, dtype=torch.float32)
    dones = torch.zeros_like(rewards)
    values = torch.zeros_like(rewards)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    recent_returns: deque = deque(maxlen=10_000)
    recent_lengths: deque = deque(maxlen=10_000)
    global_step = 0
    start_time = time.time()
    last_loss = last_q_value = last_grad_norm = float("nan")

    with progress_path.open("w", newline="", encoding="utf-8") as progress_handle:
        progress_writer = csv.DictWriter(progress_handle, fieldnames=progress_fields)
        progress_writer.writeheader()
        for iteration in range(1, args.num_iterations + 1):
            learning_rate = args.learning_rate
            if args.anneal_lr:
                learning_rate *= 1.0 - (iteration - 1.0) / args.num_iterations
                set_learning_rate(optimizer, learning_rate)

            for step in range(args.num_steps):
                global_step += args.num_envs
                observations[step] = next_observation
                dones[step] = next_done
                epsilon = linear_schedule(
                    args.start_e,
                    args.end_e,
                    args.exploration_fraction * args.total_timesteps,
                    global_step,
                )
                with torch.no_grad():
                    q_values = q_network(next_observation)
                    greedy_actions = q_values.argmax(dim=-1)
                    values[step] = q_values.gather(1, greedy_actions[:, None]).squeeze(1)
                random_actions = torch.randint(
                    0, env.action_dim, (args.num_envs,), device=device
                )
                explore = torch.rand(args.num_envs, device=device) < epsilon
                action = torch.where(explore, random_actions, greedy_actions)
                actions[step] = action
                next_observation, reward, next_done, info = env.step(action)
                rewards[step] = reward
                if info["episode_returns"].numel():
                    recent_returns.extend(info["episode_returns"].cpu().tolist())
                    recent_lengths.extend(info["episode_lengths"].cpu().tolist())

            with torch.no_grad():
                returns = torch.zeros_like(rewards)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_value = q_network(next_observation).max(dim=-1).values
                        next_nonterminal = 1.0 - next_done
                        returns[t] = rewards[t] + args.gamma * next_value * next_nonterminal
                    else:
                        next_nonterminal = 1.0 - dones[t + 1]
                        next_value = values[t + 1]
                        returns[t] = rewards[t] + args.gamma * (
                            args.q_lambda * returns[t + 1]
                            + (1.0 - args.q_lambda) * next_value
                        ) * next_nonterminal

            flat_observations = observations.reshape(-1, env.observation_dim)
            flat_actions = actions.reshape(-1)
            flat_returns = returns.reshape(-1)
            indices = np.arange(args.batch_size)
            for _ in range(args.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, args.batch_size, args.minibatch_size):
                    mb_indices = torch.as_tensor(
                        indices[start : start + args.minibatch_size],
                        device=device,
                        dtype=torch.long,
                    )
                    selected_q = q_network(flat_observations[mb_indices]).gather(
                        1, flat_actions[mb_indices, None]
                    ).squeeze(1)
                    loss = F.mse_loss(selected_q, flat_returns[mb_indices])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        q_network.parameters(), args.max_grad_norm
                    )
                    optimizer.step()
                    last_loss = float(loss.detach().item())
                    last_q_value = float(selected_q.detach().mean().item())
                    last_grad_norm = float(grad_norm.detach().item())

            if iteration % args.log_interval == 0 or iteration == args.num_iterations:
                elapsed = max(time.time() - start_time, 1.0e-9)
                mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
                mean_length = float(np.mean(recent_lengths)) if recent_lengths else float("nan")
                metrics = {
                    "charts/episodic_return_mean": mean_return,
                    "charts/episodic_length_mean": mean_length,
                    "charts/epsilon": epsilon,
                    "charts/learning_rate": learning_rate,
                    "charts/SPS": int(global_step / elapsed),
                    "losses/td_loss": last_loss,
                    "losses/q_values": last_q_value,
                    "losses/grad_norm": last_grad_norm,
                }
                log_metrics(writer, wandb_run, metrics, global_step)
                row = {
                    "global_step": global_step,
                    "iteration": iteration,
                    "learning_rate": learning_rate,
                    "epsilon": epsilon,
                    "episodic_return_mean": mean_return,
                    "episodic_length_mean": mean_length,
                    "td_loss": last_loss,
                    "q_value_mean": last_q_value,
                    "grad_norm": last_grad_norm,
                    "sps": int(global_step / elapsed),
                }
                progress_writer.writerow(row)
                progress_handle.flush()
                print(
                    f"step={global_step:,} env={args.env_id} optimizer={args.optimizer} "
                    f"return={mean_return:.4f} loss={last_loss:.6f} "
                    f"SPS={int(global_step / elapsed)}"
                )

    evaluation = greedy_evaluation(
        q_network,
        args.env_id,
        args.eval_num_envs,
        args.eval_steps,
        args.seed + 100_000,
        device,
    )
    log_metrics(
        writer,
        wandb_run,
        {
            "evaluation/greedy_return": float(evaluation["eval_greedy_return"]),
            "evaluation/reward_per_1000_steps": float(
                evaluation["eval_reward_per_1000_steps"]
            ),
            "evaluation/episodes": float(evaluation["eval_episodes"]),
        },
        global_step,
    )
    elapsed = time.time() - start_time
    summary: Dict[str, object] = {
        "status": "ok",
        "algorithm": "discrete_pqn",
        "environment": "Craftax",
        "env_id": args.env_id,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "global_step": global_step,
        "elapsed_seconds": elapsed,
        "sps": int(global_step / max(elapsed, 1.0e-9)),
        "training_recent_return": float(np.mean(recent_returns)) if recent_returns else None,
        "training_recent_length": float(np.mean(recent_lengths)) if recent_lengths else None,
        "config": config,
        **evaluation,
    }
    write_json(output_dir / "summary.json", summary)
    if args.save_model:
        torch.save(
            {
                "model_state_dict": q_network.state_dict(),
                "config": config,
                "summary": summary,
            },
            output_dir / "model.pt",
        )
    writer.close()
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "eval_greedy_return": evaluation["eval_greedy_return"],
                "eval_reward_per_1000_steps": evaluation["eval_reward_per_1000_steps"],
                "eval_episodes": evaluation["eval_episodes"],
                "sps": summary["sps"],
            }
        )
        wandb_run.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

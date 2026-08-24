#!/usr/bin/env python3
"""Parallel online PQN on the bsuite-style DeepSea exploration task.

This is the discrete-action counterpart of ``pqn_atari_envpool.py``: it uses
epsilon-greedy argmax actions, short synchronous rollouts, Q(lambda) targets,
and repeated minibatch regression without a replay buffer or target network.
The environment is implemented directly in PyTorch so thousands of DeepSea
instances can run on one GPU without a Python vector-environment bottleneck.
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
from typing import Dict, List, Optional, Sequence, Tuple

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
    exp_name: str = "pqn_deepsea"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    device: Optional[str] = None

    track: bool = True
    wandb_project_name: str = "cleanRL-deepsea-pqn"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tag: Optional[str] = None
    output_dir: Optional[str] = None
    save_model: bool = False

    deepsea_size: int = 20
    total_timesteps: int = 1_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 2048
    num_steps: int = 32
    num_minibatches: int = 4
    update_epochs: int = 4
    anneal_lr: bool = True
    gamma: float = 0.99
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    max_grad_norm: float = 10.0
    q_lambda: float = 0.65

    hidden_dim: int = 256
    hidden_layers: int = 2
    norm_type: str = "layernorm"

    optimizer: str = "Adam"
    momentum: float = 0.95
    weight_decay: float = 1.0e-4
    use_muon_input: bool = True
    use_muon_output: bool = False
    muon_ns_steps: int = 0

    eval_episodes: int = 2048
    log_interval: int = 1

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


class DeepSeaBatch:
    """A fixed-map batch of independent bsuite-style DeepSea environments."""

    def __init__(
        self,
        size: int,
        num_envs: int,
        device: torch.device,
        mapping_seed: int,
        action_map: Optional[torch.Tensor] = None,
    ) -> None:
        if size < 2:
            raise ValueError("deepsea_size must be at least 2")
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.size = int(size)
        self.num_envs = int(num_envs)
        self.device = device
        if action_map is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(mapping_seed))
            action_map = torch.randint(
                0, 2, (self.size, self.size), generator=generator
            )
        if tuple(action_map.shape) != (self.size, self.size):
            raise ValueError("action_map must have shape [deepsea_size, deepsea_size]")
        self.action_map = action_map.to(device=device, dtype=torch.long).clone()
        self.horizon = self.size - 1
        self.right_cost = 0.01 / float(self.size)
        self.rows = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.columns = torch.zeros_like(self.rows)
        self.episode_returns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        self.episode_lengths = torch.zeros_like(self.rows)

    def reset(self) -> torch.Tensor:
        self.rows.zero_()
        self.columns.zero_()
        self.episode_returns.zero_()
        self.episode_lengths.zero_()
        return self.state_indices()

    def state_indices(self) -> torch.Tensor:
        return self.rows * self.size + self.columns

    @torch.no_grad()
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        actions = actions.to(device=self.device, dtype=torch.long).reshape(-1)
        if actions.numel() != self.num_envs:
            raise ValueError("one action is required per DeepSea environment")

        right_actions = self.action_map[self.rows, self.columns]
        moved_right = actions.eq(right_actions)
        rewards = -moved_right.float() * self.right_cost
        self.columns.add_(torch.where(moved_right, 1, -1)).clamp_(0, self.size - 1)
        self.episode_lengths.add_(1)
        done = self.episode_lengths.eq(self.horizon)
        success = done & self.columns.eq(self.size - 1)
        rewards.add_(success.float())
        self.episode_returns.add_(rewards)

        completed_returns = self.episode_returns[done].clone()
        completed_success = success[done].float().clone()
        completed_lengths = self.episode_lengths[done].clone()

        self.rows.add_(1).clamp_(max=self.size - 1)
        if torch.any(done):
            self.rows[done] = 0
            self.columns[done] = 0
            self.episode_returns[done] = 0.0
            self.episode_lengths[done] = 0

        info = {
            "episode_returns": completed_returns,
            "episode_success": completed_success,
            "episode_lengths": completed_lengths,
            "moved_right": moved_right.float(),
        }
        return self.state_indices(), rewards, done.float(), info


class QNetwork(nn.Module):
    def __init__(
        self,
        deepsea_size: int,
        hidden_dim: int,
        hidden_layers: int,
        norm_type: str,
        use_muon_input: bool,
        use_muon_output: bool,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        self.deepsea_size = int(deepsea_size)
        self.num_states = self.deepsea_size**2
        self.use_muon_input = bool(use_muon_input)
        self.use_muon_output = bool(use_muon_output)
        normalized = norm_type.lower().replace("_", "")
        if normalized not in {"layernorm", "none"}:
            raise ValueError("norm_type must be 'layernorm' or 'none'")

        self.input_layer = layer_init(nn.Linear(self.num_states, hidden_dim))
        self.hidden = nn.ModuleList(
            [layer_init(nn.Linear(hidden_dim, hidden_dim)) for _ in range(hidden_layers - 1)]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(hidden_dim) if normalized == "layernorm" else nn.Identity()
                for _ in range(hidden_layers)
            ]
        )
        self.q_head = layer_init(nn.Linear(hidden_dim, 2), std=1.0)

    def features(self, state_indices: torch.Tensor) -> torch.Tensor:
        states = F.one_hot(
            state_indices.long().reshape(-1), num_classes=self.num_states
        ).float()
        x = F.relu(self.norms[0](self.input_layer(states)))
        for layer, norm in zip(self.hidden, self.norms[1:]):
            x = F.relu(norm(layer(x)))
        return x

    def forward(self, state_indices: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.features(state_indices))

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
    writer: SummaryWriter,
    wandb_run,
    metrics: Dict[str, float],
    global_step: int,
) -> None:
    for key, value in metrics.items():
        writer.add_scalar(key, value, global_step)
    if wandb_run is not None:
        wandb_run.log(metrics, step=global_step)


@torch.no_grad()
def greedy_evaluation(
    q_network: QNetwork,
    training_env: DeepSeaBatch,
    episodes: int,
) -> Dict[str, float]:
    q_network.eval()
    env = DeepSeaBatch(
        size=training_env.size,
        num_envs=max(1, episodes),
        device=training_env.device,
        mapping_seed=0,
        action_map=training_env.action_map,
    )
    states = env.reset()
    returns: List[torch.Tensor] = []
    successes: List[torch.Tensor] = []
    for _ in range(env.horizon):
        actions = q_network(states).argmax(dim=-1)
        states, _, _, info = env.step(actions)
        if info["episode_returns"].numel():
            returns.append(info["episode_returns"])
            successes.append(info["episode_success"])

    all_states = torch.arange(
        (training_env.size - 1) * training_env.size,
        device=training_env.device,
        dtype=torch.long,
    )
    predicted_actions = q_network(all_states).argmax(dim=-1)
    action_accuracy = (
        predicted_actions.eq(training_env.action_map[:-1].reshape(-1)).float().mean()
    )
    episode_returns = torch.cat(returns)
    episode_successes = torch.cat(successes)
    q_network.train()
    return {
        "eval_greedy_return": float(episode_returns.mean().item()),
        "eval_greedy_return_std": float(episode_returns.std(unbiased=False).item()),
        "eval_greedy_success": float(episode_successes.mean().item()),
        "eval_optimal_action_accuracy": float(action_accuracy.item()),
        "eval_episodes": int(episode_returns.numel()),
    }


def validate_args(args: Args) -> None:
    if args.deepsea_size < 2:
        raise ValueError("deepsea_size must be at least 2")
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


def main() -> None:
    args = tyro.cli(Args)
    validate_args(args)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    run_name = (
        f"DeepSea-{args.deepsea_size}__{args.exp_name}__{args.optimizer}__"
        f"seed{args.seed}__{int(time.time())}"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else REPO_ROOT / "logs" / "deepsea" / "runs" / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = asdict(args)
    config["resolved_device"] = str(device)
    write_json(output_dir / "config.json", config)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    wandb_run = None
    if args.track:
        import wandb

        tags = [args.wandb_tag] if args.wandb_tag else None
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            tags=tags,
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
        "success_rate",
        "td_loss",
        "q_value_mean",
        "grad_norm",
        "sps",
    ]

    env = DeepSeaBatch(args.deepsea_size, args.num_envs, device, args.seed)
    q_network = QNetwork(
        args.deepsea_size,
        args.hidden_dim,
        args.hidden_layers,
        args.norm_type,
        args.use_muon_input,
        args.use_muon_output,
    ).to(device)
    optimizer = make_optimizer(q_network, args, device)

    state_indices = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.long, device=device
    )
    actions = torch.zeros_like(state_indices)
    rewards = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.float32, device=device
    )
    dones = torch.zeros_like(rewards)
    values = torch.zeros_like(rewards)

    next_state = env.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    recent_returns: deque = deque(maxlen=10_000)
    recent_successes: deque = deque(maxlen=10_000)
    global_step = 0
    start_time = time.time()
    last_loss = float("nan")
    last_q_value = float("nan")
    last_grad_norm = float("nan")

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
                state_indices[step] = next_state
                dones[step] = next_done
                epsilon = linear_schedule(
                    args.start_e,
                    args.end_e,
                    args.exploration_fraction * args.total_timesteps,
                    global_step,
                )
                with torch.no_grad():
                    q_values = q_network(next_state)
                    greedy_actions = q_values.argmax(dim=-1)
                    values[step] = q_values.gather(1, greedy_actions[:, None]).squeeze(1)
                random_actions = torch.randint(0, 2, (args.num_envs,), device=device)
                explore = torch.rand(args.num_envs, device=device) < epsilon
                action = torch.where(explore, random_actions, greedy_actions)
                actions[step] = action
                next_state, reward, next_done, info = env.step(action)
                rewards[step] = reward
                if info["episode_returns"].numel():
                    recent_returns.extend(info["episode_returns"].cpu().tolist())
                    recent_successes.extend(info["episode_success"].cpu().tolist())

            with torch.no_grad():
                returns = torch.zeros_like(rewards)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_value = q_network(next_state).max(dim=-1).values
                        next_nonterminal = 1.0 - next_done
                        returns[t] = rewards[t] + args.gamma * next_value * next_nonterminal
                    else:
                        next_nonterminal = 1.0 - dones[t + 1]
                        next_value = values[t + 1]
                        returns[t] = rewards[t] + args.gamma * (
                            args.q_lambda * returns[t + 1]
                            + (1.0 - args.q_lambda) * next_value
                        ) * next_nonterminal

            flat_states = state_indices.reshape(-1)
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
                    selected_q = q_network(flat_states[mb_indices]).gather(
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
                success_rate = float(np.mean(recent_successes)) if recent_successes else 0.0
                metrics = {
                    "charts/episodic_return_mean": mean_return,
                    "charts/success_rate": success_rate,
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
                    "success_rate": success_rate,
                    "td_loss": last_loss,
                    "q_value_mean": last_q_value,
                    "grad_norm": last_grad_norm,
                    "sps": int(global_step / elapsed),
                }
                progress_writer.writerow(row)
                progress_handle.flush()
                print(
                    f"step={global_step:,} size={args.deepsea_size} optimizer={args.optimizer} "
                    f"return={mean_return:.4f} success={success_rate:.4f} "
                    f"loss={last_loss:.6f} SPS={int(global_step / elapsed)}"
                )

    evaluation = greedy_evaluation(q_network, env, args.eval_episodes)
    eval_metrics = {
        "evaluation/greedy_return": float(evaluation["eval_greedy_return"]),
        "evaluation/greedy_success": float(evaluation["eval_greedy_success"]),
        "evaluation/optimal_action_accuracy": float(
            evaluation["eval_optimal_action_accuracy"]
        ),
    }
    log_metrics(writer, wandb_run, eval_metrics, global_step)
    elapsed = time.time() - start_time
    summary: Dict[str, object] = {
        "status": "ok",
        "algorithm": "discrete_pqn",
        "environment": "DeepSea",
        "deepsea_size": args.deepsea_size,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "global_step": global_step,
        "elapsed_seconds": elapsed,
        "sps": int(global_step / max(elapsed, 1.0e-9)),
        "training_recent_return": float(np.mean(recent_returns)) if recent_returns else None,
        "training_recent_success": float(np.mean(recent_successes)) if recent_successes else None,
        "config": config,
        **evaluation,
    }
    write_json(output_dir / "summary.json", summary)
    if args.save_model:
        torch.save(
            {
                "model_state_dict": q_network.state_dict(),
                "action_map": env.action_map.cpu(),
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
                "eval_greedy_success": evaluation["eval_greedy_success"],
                "eval_optimal_action_accuracy": evaluation["eval_optimal_action_accuracy"],
                "sps": summary["sps"],
            }
        )
        wandb_run.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

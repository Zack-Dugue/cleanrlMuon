#!/usr/bin/env python3
"""Pure-JAX recurrent PQN for Craftax with Adam and Muon.

Craftax stepping, the Flax LSTM, replay-free Q(lambda), and every optimizer
update are composed into one outer ``jax.jit``. This follows the high-throughput
PureJaxQL execution structure while preserving this repository's matched
Adam-versus-Muon experiment and JSON/CSV/W&B launcher contract.
"""

# Keep the delayed JAX import block in place so XLA sees the CLI-derived
# platform and memory settings before backend initialization.
# ruff: noqa: I001, UP006, UP045

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple


def _early_cli_value(flag: str, default: str) -> str:
    """Read one option before importing JAX, when XLA flags still take effect."""
    arguments = sys.argv[1:]
    for index, argument in enumerate(arguments):
        if argument.startswith(flag + "="):
            return argument.split("=", 1)[1]
        if argument == flag and index + 1 < len(arguments):
            return arguments[index + 1]
    return default


def _cpu_was_requested() -> bool:
    arguments = sys.argv[1:]
    if "--no-cuda" in arguments:
        return True
    for index, argument in enumerate(arguments):
        if argument == "--device" and index + 1 < len(arguments):
            return arguments[index + 1].lower() == "cpu"
        if argument.startswith("--device="):
            return argument.split("=", 1)[1].lower() == "cpu"
    return False


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault(
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    _early_cli_value("--jax-mem-fraction", "0.90"),
)
if _cpu_was_requested():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax import core, serialization, struct, traverse_util
from flax.training import train_state
from tensorboardX import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Args:
    exp_name: str = "pqn_craftax"
    seed: int = 1
    cuda: bool = True
    device: Optional[str] = None

    track: bool = True
    wandb_project_name: str = "cleanRL-craftax-pqn"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tag: Optional[str] = None
    output_dir: Optional[str] = None
    save_model: bool = False

    env_id: str = "Craftax-Symbolic-v1"
    total_timesteps: int = 1_000_000_000
    learning_rate: float = 3.0e-4
    num_envs: int = 1024
    num_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    anneal_lr: bool = True
    gamma: float = 0.99
    start_e: float = 1.0
    end_e: float = 0.005
    exploration_fraction: float = 0.10
    max_grad_norm: float = 0.5
    q_lambda: float = 0.50

    hidden_dim: int = 512
    hidden_layers: int = 1
    norm_type: str = "layernorm"
    norm_input: bool = True
    add_last_action: bool = True
    batch_renorm_momentum: float = 0.01
    batch_renorm_eps: float = 1.0e-5
    batch_renorm_max_r: float = 3.0
    batch_renorm_max_d: float = 5.0
    batch_renorm_warmup_steps: int = 10_000
    batch_renorm_smooth: bool = True
    optimistic_resets: bool = True
    optimistic_reset_ratio: int = 16
    compile_model: bool = True

    optimizer: str = "Adam"
    momentum: float = 0.95
    weight_decay: float = 1.0e-4
    use_muon_input: bool = True
    use_muon_output: bool = False
    muon_ns_steps: int = 0

    # Vector steps per evaluation environment, matching TEST_NUM_STEPS.
    eval_steps: int = 10_000
    eval_num_envs: int = 512
    log_interval: int = 1
    matrix_diagnostics_interval: int = 100
    matrix_diagnostics_power_iterations: int = 8
    jax_mem_fraction: float = 0.90

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class SoftBatchRenorm(nn.Module):
    """TorchRL-style BatchRenorm with a smooth BatchNorm-to-renorm ramp."""

    momentum: float = 0.01
    eps: float = 1.0e-5
    max_r: float = 3.0
    max_d: float = 5.0
    warmup_steps: int = 10_000
    smooth: bool = True

    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        features = x.shape[-1]
        running_mean = self.variable(
            "batch_stats", "mean", lambda: jnp.zeros((features,), jnp.float32)
        )
        running_var = self.variable(
            "batch_stats", "var", lambda: jnp.ones((features,), jnp.float32)
        )
        count = self.variable("batch_stats", "count", lambda: jnp.zeros((), jnp.int32))
        scale = self.param("scale", nn.initializers.ones_init(), (features,))
        bias = self.param("bias", nn.initializers.zeros_init(), (features,))
        running_std = jnp.sqrt(running_var.value + self.eps)

        if train:
            reduction_axes = tuple(range(x.ndim - 1))
            batch_mean = jnp.mean(x, axis=reduction_axes)
            batch_var = jnp.var(x, axis=reduction_axes)
            batch_std = jnp.sqrt(batch_var + self.eps)
            r = jnp.clip(
                jax.lax.stop_gradient(batch_std / running_std),
                1.0 / self.max_r,
                self.max_r,
            )
            d = jnp.clip(
                jax.lax.stop_gradient((batch_mean - running_mean.value) / running_std),
                -self.max_d,
                self.max_d,
            )
            if self.warmup_steps > 0:
                if self.smooth:
                    factor = jnp.minimum(
                        count.value.astype(x.dtype) / self.warmup_steps, 1.0
                    )
                else:
                    factor = (count.value >= self.warmup_steps).astype(x.dtype)
                r = 1.0 + (r - 1.0) * factor
                d = d * factor

            normalized = (x - batch_mean) / batch_std * r + d
            sample_count = int(np.prod(x.shape[:-1]))
            if sample_count > 1:
                unbiased_var = batch_var * sample_count / (sample_count - 1)
            else:
                unbiased_var = batch_var
            running_mean.value = running_mean.value + self.momentum * (
                batch_mean - running_mean.value
            )
            running_var.value = running_var.value + self.momentum * (
                unbiased_var - running_var.value
            )
            count.value = jnp.minimum(count.value + 1, max(self.warmup_steps, 1))
        else:
            normalized = (x - running_mean.value) / running_std
        return normalized * scale + bias


def _block_orthogonal_lstm_init(
    key: jax.Array, shape: Tuple[int, ...], dtype: Any = jnp.float32
) -> jax.Array:
    """Four independent orthogonal recurrent gate matrices."""
    if len(shape) != 2 or shape[1] % 4:
        raise ValueError(f"Expected recurrent shape [hidden, 4*hidden], got {shape}")
    gate_width = shape[1] // 4
    keys = jax.random.split(key, 4)
    initializer = nn.initializers.orthogonal()
    blocks = [initializer(gate_key, (shape[0], gate_width), dtype) for gate_key in keys]
    return jnp.concatenate(blocks, axis=1)


class ResetLSTMCell(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(
        self,
        carry: Tuple[jax.Array, jax.Array],
        inputs: Tuple[jax.Array, jax.Array],
    ) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
        x, reset = inputs
        cell, hidden = carry
        keep = jnp.logical_not(reset)[..., None]
        cell = jnp.where(keep, cell, jnp.zeros_like(cell))
        hidden = jnp.where(keep, hidden, jnp.zeros_like(hidden))

        input_kernel = self.param(
            "input_kernel",
            nn.initializers.xavier_uniform(),
            (x.shape[-1], 4 * self.hidden_dim),
        )
        recurrent_kernel = self.param(
            "recurrent_kernel",
            _block_orthogonal_lstm_init,
            (self.hidden_dim, 4 * self.hidden_dim),
        )
        bias = self.param("bias", nn.initializers.zeros_init(), (4 * self.hidden_dim,))
        gates = x @ input_kernel + hidden @ recurrent_kernel + bias
        input_gate, forget_gate, candidate, output_gate = jnp.split(gates, 4, axis=-1)
        input_gate = jax.nn.sigmoid(input_gate)
        forget_gate = jax.nn.sigmoid(forget_gate)
        candidate = jnp.tanh(candidate)
        output_gate = jax.nn.sigmoid(output_gate)
        new_cell = forget_gate * cell + input_gate * candidate
        new_hidden = output_gate * jnp.tanh(new_cell)
        return (new_cell, new_hidden), new_hidden


class RNNQNetwork(nn.Module):
    observation_dim: int
    action_dim: int
    hidden_dim: int = 512
    hidden_layers: int = 1
    norm_type: str = "layernorm"
    norm_input: bool = True
    add_last_action: bool = True
    batch_renorm_momentum: float = 0.01
    batch_renorm_eps: float = 1.0e-5
    batch_renorm_max_r: float = 3.0
    batch_renorm_max_d: float = 5.0
    batch_renorm_warmup_steps: int = 10_000
    batch_renorm_smooth: bool = True

    @nn.compact
    def __call__(
        self,
        carry: Tuple[jax.Array, jax.Array],
        observations: jax.Array,
        last_dones: jax.Array,
        last_actions: jax.Array,
        *,
        train: bool,
    ) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
        x = observations.astype(jnp.float32).reshape(
            *observations.shape[:-1], self.observation_dim
        )
        if self.norm_input:
            x = SoftBatchRenorm(
                momentum=self.batch_renorm_momentum,
                eps=self.batch_renorm_eps,
                max_r=self.batch_renorm_max_r,
                max_d=self.batch_renorm_max_d,
                warmup_steps=self.batch_renorm_warmup_steps,
                smooth=self.batch_renorm_smooth,
                name="input_renorm",
            )(x, train=train)

        normalized = self.norm_type.lower().replace("_", "")
        for layer_index in range(self.hidden_layers):
            x = nn.Dense(self.hidden_dim, name=f"encoder_{layer_index}")(x)
            if normalized == "layernorm":
                x = nn.LayerNorm(name=f"encoder_norm_{layer_index}")(x)
            x = nn.relu(x)

        if self.add_last_action:
            previous_action = jax.nn.one_hot(last_actions, self.action_dim)
            x = jnp.concatenate((x, previous_action), axis=-1)

        scanned_cell = nn.scan(
            ResetLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        carry, x = scanned_cell(self.hidden_dim, name="lstm")(carry, (x, last_dones))
        q_values = nn.Dense(self.action_dim, name="q_head")(x)
        return carry, q_values

    def initialize_carry(self, batch_size: int) -> Tuple[jax.Array, jax.Array]:
        shape = (int(batch_size), self.hidden_dim)
        return jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32)


class RLMuonState(NamedTuple):
    momentum: Any


def orthogonalize_newton_schulz(gradient: jax.Array, steps: int) -> jax.Array:
    """Match the repository's bfloat16 quintic Newton--Schulz Muon step."""
    if gradient.ndim != 2:
        raise ValueError("Muon requires matrix-valued parameter gradients")
    original_dtype = gradient.dtype
    x = gradient.astype(jnp.bfloat16)
    transposed = gradient.shape[0] > gradient.shape[1]
    if transposed:
        x = x.T
    norm = jnp.linalg.norm(x.astype(jnp.float32)).astype(x.dtype)
    x = x / (norm + jnp.asarray(1.0e-7, x.dtype))

    def iteration(_: int, current: jax.Array) -> jax.Array:
        gram = current @ current.T
        correction = -4.7750 * gram + 2.0315 * (gram @ gram)
        return 3.4445 * current + correction @ current

    x = jax.lax.fori_loop(0, steps, iteration, x)
    if transposed:
        x = x.T
    return x.astype(original_dtype)


def scale_by_rl_muon(
    *, momentum: float, ns_steps: int, nesterov: bool = True
) -> optax.GradientTransformation:
    """Muon direction with the same momentum and width scaling as this repo."""

    def init_fn(params: Any) -> RLMuonState:
        return RLMuonState(momentum=jax.tree_util.tree_map(jnp.zeros_like, params))

    def update_fn(
        updates: Any, state: RLMuonState, params: Any = None
    ) -> Tuple[Any, RLMuonState]:
        del params
        new_momentum = jax.tree_util.tree_map(
            lambda old, grad: momentum * old + (1.0 - momentum) * grad,
            state.momentum,
            updates,
        )
        if nesterov:
            directions = jax.tree_util.tree_map(
                lambda grad, moving: (1.0 - momentum) * grad + momentum * moving,
                updates,
                new_momentum,
            )
        else:
            directions = new_momentum
        directions = jax.tree_util.tree_map(
            lambda value: (
                orthogonalize_newton_schulz(value, ns_steps)
                * (0.2 * np.sqrt(max(value.shape)))
            ),
            directions,
        )
        return directions, RLMuonState(momentum=new_momentum)

    return optax.GradientTransformation(init_fn, update_fn)


def parameter_labels(
    params: Any, *, use_muon_input: bool, use_muon_output: bool
) -> Any:
    """Route the same encoder/LSTM/output matrices as the PyTorch experiment."""
    flat_params = traverse_util.flatten_dict(params)
    flat_labels: Dict[Tuple[str, ...], str] = {}
    for path, value in flat_params.items():
        joined = "/".join(path)
        label = "adam"
        if value.ndim == 2:
            if "q_head" in joined:
                label = "muon" if use_muon_output else "adam"
            elif "encoder_0" in joined:
                label = "muon" if use_muon_input else "adam"
            else:
                label = "muon"
        flat_labels[path] = label
    labels = traverse_util.unflatten_dict(flat_labels)
    return core.freeze(labels) if isinstance(params, core.FrozenDict) else labels


def make_learning_rate(args: Args):
    if not args.anneal_lr:
        return args.learning_rate
    total_gradient_steps = (
        args.num_iterations * args.update_epochs * args.num_minibatches
    )
    return optax.linear_schedule(
        init_value=args.learning_rate,
        end_value=0.0,
        transition_steps=max(total_gradient_steps, 1),
    )


def make_optimizer(
    params: Any, args: Args, *, ns_steps: int
) -> optax.GradientTransformation:
    learning_rate = make_learning_rate(args)
    if args.optimizer.lower() == "adam":
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            b1=args.momentum,
            b2=0.99,
            eps=1.0e-5,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer.lower() == "muon":
        labels = parameter_labels(
            params,
            use_muon_input=args.use_muon_input,
            use_muon_output=args.use_muon_output,
        )
        muon_learning_rate = (
            optax.scale_by_schedule(learning_rate)
            if callable(learning_rate)
            else optax.scale(learning_rate)
        )
        optimizer = optax.multi_transform(
            {
                "muon": optax.chain(
                    scale_by_rl_muon(
                        momentum=args.momentum,
                        ns_steps=ns_steps,
                        nesterov=True,
                    ),
                    optax.add_decayed_weights(args.weight_decay),
                    muon_learning_rate,
                    optax.scale(-1.0),
                ),
                "adam": optax.adamw(
                    learning_rate=learning_rate,
                    b1=args.momentum,
                    b2=0.99,
                    eps=1.0e-5,
                    weight_decay=args.weight_decay,
                ),
            },
            labels,
        )
    else:
        raise ValueError("optimizer must be Adam or Muon")
    return optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optimizer)


def _find_muon_state(value: Any) -> RLMuonState:
    """Find the custom Muon state inside Optax chain/masking containers."""
    if isinstance(value, RLMuonState):
        return value
    if isinstance(value, Mapping):
        children = value.values()
    elif isinstance(value, (tuple, list)):
        children = value
    else:
        children = ()
    matches = []
    for child in children:
        try:
            matches.append(_find_muon_state(child))
        except LookupError:
            pass
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError("More than one RLMuonState was found in the Optax state")
    raise LookupError("RLMuonState was not found in the Optax state")


def _matrix_slug(path: Tuple[str, ...]) -> str:
    return ".".join(str(part) for part in path)


def _stable_rank_statistics(
    matrix: jax.Array, *, power_iterations: int
) -> Dict[str, jax.Array]:
    """Cheap scale-invariant stable-rank estimate without an SVD.

    Power iteration is applied implicitly to the smaller-side Gram matrix, so
    this never materializes the large observation-encoder Gram. The estimate
    is clipped to the mathematically valid [0, min(m, n)] interval.
    """
    value = matrix.astype(jnp.float32)
    rows, columns = value.shape
    rank_limit = float(min(rows, columns))
    frobenius_squared = jnp.sum(jnp.square(value))
    frobenius_norm = jnp.sqrt(frobenius_squared)
    safe_frobenius = jnp.maximum(frobenius_norm, 1.0e-15)
    normalized = value / safe_frobenius
    probe_size = min(rows, columns)
    probe = jnp.sin(jnp.arange(1, probe_size + 1, dtype=jnp.float32))
    probe = probe / jnp.maximum(jnp.linalg.norm(probe), 1.0e-12)

    def gram_product(vector: jax.Array) -> jax.Array:
        if rows <= columns:
            return normalized @ (normalized.T @ vector)
        return normalized.T @ (normalized @ vector)

    def power_step(_: int, vector: jax.Array) -> jax.Array:
        product = gram_product(vector)
        return product / jnp.maximum(jnp.linalg.norm(product), 1.0e-12)

    probe = jax.lax.fori_loop(0, power_iterations, power_step, probe)
    largest_normalized_eigenvalue = jnp.maximum(
        jnp.vdot(probe, gram_product(probe)).real, 1.0e-12
    )
    stable_rank = jnp.where(
        frobenius_squared > 0.0,
        jnp.clip(1.0 / largest_normalized_eigenvalue, 0.0, rank_limit),
        0.0,
    )
    return {
        "stable_rank": stable_rank,
        "stable_rank_fraction": stable_rank / rank_limit,
        "frobenius_norm": frobenius_norm,
        "spectral_norm_estimate": jnp.sqrt(
            largest_normalized_eigenvalue * frobenius_squared
        ),
        "rms": jnp.sqrt(frobenius_squared / float(matrix.size)),
        "max_abs": jnp.max(jnp.abs(value)),
    }


def _matrix_diagnostics(
    matrices: Mapping[Tuple[str, ...], jax.Array],
    *,
    prefix: str,
    power_iterations: int,
    include_scale: bool,
) -> Dict[str, jax.Array]:
    diagnostics: Dict[str, jax.Array] = {}
    for path, matrix in matrices.items():
        if getattr(matrix, "ndim", 0) != 2:
            continue
        statistics = _stable_rank_statistics(
            matrix, power_iterations=power_iterations
        )
        name = _matrix_slug(path)
        diagnostics[f"diagnostics/{prefix}/{name}/stable_rank"] = statistics[
            "stable_rank"
        ]
        diagnostics[f"diagnostics/{prefix}/{name}/stable_rank_fraction"] = (
            statistics["stable_rank_fraction"]
        )
        diagnostics[f"diagnostics/{prefix}/{name}/frobenius_norm"] = statistics[
            "frobenius_norm"
        ]
        diagnostics[f"diagnostics/{prefix}/{name}/spectral_norm_estimate"] = (
            statistics["spectral_norm_estimate"]
        )
        if include_scale:
            diagnostics[f"diagnostics/{prefix}/{name}/rms"] = statistics["rms"]
            diagnostics[f"diagnostics/{prefix}/{name}/max_abs"] = statistics[
                "max_abs"
            ]
    return diagnostics


class TrainState(train_state.TrainState):
    batch_stats: Any


@struct.dataclass
class Transition:
    observation: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    last_done: jax.Array
    last_action: jax.Array


@struct.dataclass
class EpisodeEvent:
    episode_return: jax.Array
    episode_length: jax.Array
    returned_episode: jax.Array


def make_vector_environment(
    env: Any,
    env_params: Any,
    *,
    num_envs: int,
    optimistic_resets: bool,
    optimistic_reset_ratio: int,
):
    reset_one = jax.vmap(env.reset, in_axes=(0, None))
    step_one = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def reset(key: jax.Array):
        return reset_one(jax.random.split(key, num_envs), env_params)

    if optimistic_resets:
        reset_ratio = min(optimistic_reset_ratio, num_envs)
        if num_envs % reset_ratio:
            raise ValueError("optimistic_reset_ratio must exactly divide num_envs")
        num_resets = num_envs // reset_ratio

        def step(key: jax.Array, state: Any, action: jax.Array):
            step_key, reset_key, choice_key = jax.random.split(key, 3)
            obs_step, state_step, reward, done, _ = step_one(
                jax.random.split(step_key, num_envs), state, action, env_params
            )
            obs_reset, state_reset = reset_one(
                jax.random.split(reset_key, num_resets), env_params
            )
            reset_indexes = jnp.arange(num_resets).repeat(reset_ratio)
            being_reset = jax.random.choice(
                choice_key,
                jnp.arange(num_envs),
                shape=(num_resets,),
                p=done,
                replace=False,
            )
            reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(num_resets))
            obs_reset = obs_reset[reset_indexes]
            state_reset = jax.tree_util.tree_map(
                lambda value: value[reset_indexes], state_reset
            )
            observation = jax.vmap(jax.lax.select)(done, obs_reset, obs_step)
            state = jax.tree_util.tree_map(
                lambda reset_value, step_value: jax.vmap(jax.lax.select)(
                    done, reset_value, step_value
                ),
                state_reset,
                state_step,
            )
            return observation, state, reward, done

    else:

        def step(key: jax.Array, state: Any, action: jax.Array):
            observation, state, reward, done, _ = step_one(
                jax.random.split(key, num_envs), state, action, env_params
            )
            return observation, state, reward, done

    return reset, step


def q_lambda_targets(
    rewards: jax.Array,
    dones: jax.Array,
    max_q_values: jax.Array,
    *,
    gamma: float,
    q_lambda: float,
) -> jax.Array:
    """The first time-1 replay-free truncated Q(lambda) targets."""
    final_target = rewards[-2] + gamma * (1.0 - dones[-2]) * max_q_values[-1]

    def backward_step(next_target: jax.Array, values: Tuple[jax.Array, ...]):
        reward, done, next_q = values
        target = reward + gamma * (1.0 - done) * (
            q_lambda * next_target + (1.0 - q_lambda) * next_q
        )
        return target, target

    _, prefix = jax.lax.scan(
        backward_step,
        final_target,
        (rewards[:-2], dones[:-2], max_q_values[1:-1]),
        reverse=True,
    )
    return jnp.concatenate((prefix, final_target[None]), axis=0)


def make_train_function(
    args: Args,
    network: RNNQNetwork,
    env_reset: Any,
    env_step: Any,
    action_dim: int,
):
    envs_per_minibatch = args.num_envs // args.num_minibatches
    epsilon_schedule = optax.linear_schedule(
        init_value=args.start_e,
        end_value=args.end_e,
        transition_steps=max(int(args.exploration_fraction * args.num_iterations), 1),
    )
    learning_rate = make_learning_rate(args)

    def train(initial_train_state: TrainState, rng: jax.Array):
        initial_flat_params = traverse_util.flatten_dict(initial_train_state.params)
        matrix_paths = tuple(
            path for path, value in initial_flat_params.items() if value.ndim == 2
        )
        flat_muon_labels = traverse_util.flatten_dict(
            parameter_labels(
                initial_train_state.params,
                use_muon_input=args.use_muon_input,
                use_muon_output=args.use_muon_output,
            )
        )
        muon_matrix_paths = tuple(
            path for path in matrix_paths if flat_muon_labels[path] == "muon"
        )

        diagnostic_keys = []
        for prefix, paths, include_scale in (
            ("weights", matrix_paths, True),
            ("raw_gradient", matrix_paths, False),
            (
                "pre_ns_momentum",
                muon_matrix_paths if args.optimizer.lower() == "muon" else (),
                False,
            ),
        ):
            for path in paths:
                name = _matrix_slug(path)
                diagnostic_keys.extend(
                    (
                        f"diagnostics/{prefix}/{name}/stable_rank",
                        f"diagnostics/{prefix}/{name}/stable_rank_fraction",
                        f"diagnostics/{prefix}/{name}/frobenius_norm",
                        f"diagnostics/{prefix}/{name}/spectral_norm_estimate",
                    )
                )
                if include_scale:
                    diagnostic_keys.extend(
                        (
                            f"diagnostics/{prefix}/{name}/rms",
                            f"diagnostics/{prefix}/{name}/max_abs",
                        )
                    )
        diagnostic_keys = tuple(diagnostic_keys)

        rng, reset_key = jax.random.split(rng)
        observation, env_state = env_reset(reset_key)
        rollout_state = (
            network.initialize_carry(args.num_envs),
            observation,
            jnp.zeros((args.num_envs,), jnp.bool_),
            jnp.zeros((args.num_envs,), jnp.int32),
            env_state,
            jnp.zeros((args.num_envs,), jnp.float32),
            jnp.zeros((args.num_envs,), jnp.int32),
        )

        def update_step(runner: Tuple[Any, ...], update_index: jax.Array):
            train_state_value, rollout_state_value, update_rng = runner
            epsilon = epsilon_schedule(update_index)
            initial_carry = rollout_state_value[0]
            diagnostics_due = jnp.logical_or(
                jnp.logical_or(
                    update_index == 0,
                    (update_index + 1) % args.matrix_diagnostics_interval == 0,
                ),
                update_index == args.num_iterations - 1,
            )

            def environment_step(state: Tuple[Any, ...], _: None):
                (
                    carry,
                    last_observation,
                    last_done,
                    last_action,
                    current_env_state,
                    running_return,
                    running_length,
                    step_rng,
                ) = state
                step_rng, action_key, explore_key, env_key = jax.random.split(
                    step_rng, 4
                )
                new_carry, q_values = network.apply(
                    {
                        "params": train_state_value.params,
                        "batch_stats": train_state_value.batch_stats,
                    },
                    carry,
                    last_observation[None],
                    last_done[None],
                    last_action[None],
                    train=False,
                )
                q_values = q_values[0]
                greedy_action = jnp.argmax(q_values, axis=-1)
                random_action = jax.random.randint(
                    action_key,
                    (args.num_envs,),
                    minval=0,
                    maxval=action_dim,
                )
                action = jnp.where(
                    jax.random.uniform(explore_key, (args.num_envs,)) < epsilon,
                    random_action,
                    greedy_action,
                )
                new_observation, new_env_state, reward, done = env_step(
                    env_key, current_env_state, action
                )
                completed_return = running_return + reward
                completed_length = running_length + 1
                event = EpisodeEvent(
                    episode_return=jnp.where(done, completed_return, 0.0),
                    episode_length=jnp.where(done, completed_length, 0),
                    returned_episode=done,
                )
                transition = Transition(
                    observation=last_observation,
                    action=action,
                    reward=reward,
                    done=done,
                    last_done=last_done,
                    last_action=last_action,
                )
                new_state = (
                    new_carry,
                    new_observation,
                    done,
                    action,
                    new_env_state,
                    jnp.where(done, 0.0, completed_return),
                    jnp.where(done, 0, completed_length),
                    step_rng,
                )
                return new_state, (transition, event)

            update_rng, rollout_rng = jax.random.split(update_rng)
            scan_state = (*rollout_state_value, rollout_rng)
            scan_state, (transitions, events) = jax.lax.scan(
                environment_step, scan_state, None, length=args.num_steps
            )
            rollout_state_value = scan_state[:-1]

            def learn_minibatch(
                current_train_state: TrainState,
                minibatch: Tuple[jax.Array, jax.Array, Any, Transition],
            ):
                (
                    epoch_index,
                    minibatch_index,
                    minibatch_carry,
                    minibatch_transition,
                ) = minibatch

                def loss_fn(params: Any):
                    (unused_carry, q_values), updates = network.apply(
                        {
                            "params": params,
                            "batch_stats": current_train_state.batch_stats,
                        },
                        minibatch_carry,
                        minibatch_transition.observation,
                        minibatch_transition.last_done,
                        minibatch_transition.last_action,
                        train=True,
                        mutable=["batch_stats"],
                    )
                    del unused_carry
                    max_q_values = jax.lax.stop_gradient(q_values).max(axis=-1)
                    targets = q_lambda_targets(
                        minibatch_transition.reward,
                        minibatch_transition.done,
                        max_q_values,
                        gamma=args.gamma,
                        q_lambda=args.q_lambda,
                    )
                    selected_q = jnp.take_along_axis(
                        q_values[:-1],
                        minibatch_transition.action[:-1, :, None],
                        axis=-1,
                    ).squeeze(-1)
                    td_error = selected_q - targets
                    loss = 0.5 * jnp.mean(jnp.square(td_error))
                    return loss, (
                        updates["batch_stats"],
                        jnp.mean(selected_q),
                        jnp.mean(jnp.abs(q_values)),
                        jnp.max(jnp.abs(q_values)),
                        jnp.mean(targets),
                        jnp.mean(jnp.abs(targets)),
                        jnp.max(jnp.abs(targets)),
                        jnp.mean(jnp.abs(td_error)),
                        jnp.max(jnp.abs(td_error)),
                    )

                (
                    loss,
                    (
                        batch_stats,
                        q_mean,
                        q_abs_mean,
                        q_abs_max,
                        target_mean,
                        target_abs_mean,
                        target_abs_max,
                        td_error_abs_mean,
                        td_error_abs_max,
                    ),
                ), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
                    current_train_state.params
                )
                grad_norm = optax.global_norm(gradients)
                old_opt_state = current_train_state.opt_state
                current_train_state = current_train_state.apply_gradients(
                    grads=gradients, batch_stats=batch_stats
                )

                collect_diagnostics = jnp.logical_and(
                    diagnostics_due,
                    jnp.logical_and(
                        minibatch_index == args.num_minibatches - 1,
                        epoch_index == args.update_epochs - 1,
                    ),
                )

                def calculate_diagnostics(_: None) -> Dict[str, jax.Array]:
                    flat_params = traverse_util.flatten_dict(
                        current_train_state.params
                    )
                    flat_gradients = traverse_util.flatten_dict(gradients)
                    result = _matrix_diagnostics(
                        {path: flat_params[path] for path in matrix_paths},
                        prefix="weights",
                        power_iterations=args.matrix_diagnostics_power_iterations,
                        include_scale=True,
                    )
                    result.update(
                        _matrix_diagnostics(
                            {path: flat_gradients[path] for path in matrix_paths},
                            prefix="raw_gradient",
                            power_iterations=args.matrix_diagnostics_power_iterations,
                            include_scale=False,
                        )
                    )
                    if args.optimizer.lower() == "muon":
                        old_muon_state = _find_muon_state(old_opt_state)
                        flat_momentum = traverse_util.flatten_dict(
                            old_muon_state.momentum
                        )
                        clip_scale = jnp.minimum(
                            1.0,
                            args.max_grad_norm / jnp.maximum(grad_norm, 1.0e-12),
                        )
                        pre_ns_directions = {}
                        for path in muon_matrix_paths:
                            clipped_gradient = flat_gradients[path] * clip_scale
                            new_momentum = (
                                args.momentum * flat_momentum[path]
                                + (1.0 - args.momentum) * clipped_gradient
                            )
                            pre_ns_directions[path] = (
                                (1.0 - args.momentum) * clipped_gradient
                                + args.momentum * new_momentum
                            )
                        result.update(
                            _matrix_diagnostics(
                                pre_ns_directions,
                                prefix="pre_ns_momentum",
                                power_iterations=(
                                    args.matrix_diagnostics_power_iterations
                                ),
                                include_scale=False,
                            )
                        )
                    return result

                diagnostics = jax.lax.cond(
                    collect_diagnostics,
                    calculate_diagnostics,
                    lambda _: {
                        key: jnp.zeros((), jnp.float32) for key in diagnostic_keys
                    },
                    operand=None,
                )
                minibatch_metrics = {
                    "td_loss": loss,
                    "q_value_mean": q_mean,
                    "q_value_abs_mean": q_abs_mean,
                    "q_value_abs_max": q_abs_max,
                    "target_mean": target_mean,
                    "target_abs_mean": target_abs_mean,
                    "target_abs_max": target_abs_max,
                    "td_error_abs_mean": td_error_abs_mean,
                    "td_error_abs_max": td_error_abs_max,
                    "grad_norm": grad_norm,
                    "grad_clipped": (grad_norm > args.max_grad_norm).astype(
                        jnp.float32
                    ),
                }
                minibatch_metrics.update(diagnostics)
                return current_train_state, minibatch_metrics

            def learn_epoch(
                epoch_state: Tuple[TrainState, jax.Array], epoch_index: jax.Array
            ):
                current_train_state, epoch_rng = epoch_state
                epoch_rng, permutation_key = jax.random.split(epoch_rng)
                permutation = jax.random.permutation(permutation_key, args.num_envs)

                def transition_minibatches(value: jax.Array) -> jax.Array:
                    value = value[:, permutation]
                    value = value.reshape(
                        args.num_steps,
                        args.num_minibatches,
                        envs_per_minibatch,
                        *value.shape[2:],
                    )
                    return jnp.swapaxes(value, 0, 1)

                def carry_minibatches(value: jax.Array) -> jax.Array:
                    return value[permutation].reshape(
                        args.num_minibatches,
                        envs_per_minibatch,
                        *value.shape[1:],
                    )

                minibatch_transitions = jax.tree_util.tree_map(
                    transition_minibatches, transitions
                )
                minibatch_carries = jax.tree_util.tree_map(
                    carry_minibatches, initial_carry
                )
                current_train_state, minibatch_metrics = jax.lax.scan(
                    learn_minibatch,
                    current_train_state,
                    (
                        jnp.full(
                            (args.num_minibatches,), epoch_index, dtype=jnp.int32
                        ),
                        jnp.arange(args.num_minibatches, dtype=jnp.int32),
                        minibatch_carries,
                        minibatch_transitions,
                    ),
                )
                return (
                    current_train_state,
                    epoch_rng,
                ), minibatch_metrics

            update_rng, learning_rng = jax.random.split(update_rng)
            (train_state_value, _), learning_metrics = jax.lax.scan(
                learn_epoch,
                (train_state_value, learning_rng),
                jnp.arange(args.update_epochs, dtype=jnp.int32),
            )
            metric = {
                "td_loss": jnp.mean(learning_metrics["td_loss"]),
                "q_value_mean": jnp.mean(learning_metrics["q_value_mean"]),
                "q_value_abs_mean": jnp.mean(
                    learning_metrics["q_value_abs_mean"]
                ),
                "q_value_abs_max": jnp.max(learning_metrics["q_value_abs_max"]),
                "target_mean": jnp.mean(learning_metrics["target_mean"]),
                "target_abs_mean": jnp.mean(
                    learning_metrics["target_abs_mean"]
                ),
                "target_abs_max": jnp.max(learning_metrics["target_abs_max"]),
                "td_error_abs_mean": jnp.mean(
                    learning_metrics["td_error_abs_mean"]
                ),
                "td_error_abs_max": jnp.max(
                    learning_metrics["td_error_abs_max"]
                ),
                "grad_norm": jnp.mean(learning_metrics["grad_norm"]),
                "grad_norm_max": jnp.max(learning_metrics["grad_norm"]),
                "grad_clip_fraction": jnp.mean(learning_metrics["grad_clipped"]),
                "episode_return_sum": jnp.sum(events.episode_return),
                "episode_length_sum": jnp.sum(events.episode_length),
                "episode_count": jnp.sum(events.returned_episode),
                "epsilon": epsilon,
                "learning_rate": (
                    learning_rate(train_state_value.step)
                    if callable(learning_rate)
                    else jnp.asarray(learning_rate)
                ),
            }
            for key in diagnostic_keys:
                metric[key] = jnp.where(
                    diagnostics_due,
                    jnp.sum(learning_metrics[key]),
                    jnp.asarray(jnp.nan, jnp.float32),
                )
            renorm_stats = train_state_value.batch_stats["input_renorm"]
            metric.update(
                {
                    "diagnostics/batch_renorm/count": renorm_stats["count"],
                    "diagnostics/batch_renorm/ramp_fraction": jnp.minimum(
                        renorm_stats["count"].astype(jnp.float32)
                        / max(args.batch_renorm_warmup_steps, 1),
                        1.0,
                    ),
                    "diagnostics/batch_renorm/running_mean_rms": jnp.sqrt(
                        jnp.mean(jnp.square(renorm_stats["mean"]))
                    ),
                    "diagnostics/batch_renorm/running_var_mean": jnp.mean(
                        renorm_stats["var"]
                    ),
                    "diagnostics/batch_renorm/running_var_min": jnp.min(
                        renorm_stats["var"]
                    ),
                    "diagnostics/batch_renorm/running_var_max": jnp.max(
                        renorm_stats["var"]
                    ),
                }
            )
            return (
                train_state_value,
                rollout_state_value,
                update_rng,
            ), metric

        update_indexes = jnp.arange(args.num_iterations, dtype=jnp.int32)
        (final_train_state, _, _), metrics = jax.lax.scan(
            update_step,
            (initial_train_state, rollout_state, rng),
            update_indexes,
        )
        return final_train_state, metrics

    return train


def make_evaluation_function(
    args: Args,
    network: RNNQNetwork,
    env_reset: Any,
    env_step: Any,
):
    def evaluate(params: Any, batch_stats: Any, rng: jax.Array):
        rng, reset_key = jax.random.split(rng)
        observation, env_state = env_reset(reset_key)
        initial_state = (
            network.initialize_carry(args.eval_num_envs),
            observation,
            jnp.zeros((args.eval_num_envs,), jnp.bool_),
            jnp.zeros((args.eval_num_envs,), jnp.int32),
            env_state,
            jnp.zeros((args.eval_num_envs,), jnp.float32),
            jnp.zeros((args.eval_num_envs,), jnp.int32),
            jnp.zeros((args.eval_num_envs,), jnp.float32),
            jnp.zeros((), jnp.float32),
            jnp.zeros((), jnp.float32),
            jnp.zeros((), jnp.float32),
            jnp.zeros((), jnp.int32),
            rng,
        )

        def evaluation_step(state: Tuple[Any, ...], _: None):
            (
                carry,
                last_observation,
                last_done,
                last_action,
                current_env_state,
                running_return,
                running_length,
                reward_by_env,
                return_sum,
                return_square_sum,
                length_sum,
                episode_count,
                step_rng,
            ) = state
            step_rng, env_key = jax.random.split(step_rng)
            new_carry, q_values = network.apply(
                {"params": params, "batch_stats": batch_stats},
                carry,
                last_observation[None],
                last_done[None],
                last_action[None],
                train=False,
            )
            action = jnp.argmax(q_values[0], axis=-1)
            new_observation, new_env_state, reward, done = env_step(
                env_key, current_env_state, action
            )
            completed_return = running_return + reward
            completed_length = running_length + 1
            event_return = jnp.where(done, completed_return, 0.0)
            event_length = jnp.where(done, completed_length, 0)
            return (
                new_carry,
                new_observation,
                done,
                action,
                new_env_state,
                jnp.where(done, 0.0, completed_return),
                jnp.where(done, 0, completed_length),
                reward_by_env + reward,
                return_sum + jnp.sum(event_return),
                return_square_sum + jnp.sum(jnp.square(event_return)),
                length_sum + jnp.sum(event_length),
                episode_count + jnp.sum(done),
                step_rng,
            ), None

        final_state, _ = jax.lax.scan(
            evaluation_step, initial_state, None, length=args.eval_steps
        )
        reward_by_env = final_state[7]
        return_sum = final_state[8]
        return_square_sum = final_state[9]
        length_sum = final_state[10]
        episode_count = final_state[11]
        safe_count = jnp.maximum(episode_count, 1)
        completed_mean = return_sum / safe_count
        completed_variance = jnp.maximum(
            return_square_sum / safe_count - jnp.square(completed_mean), 0.0
        )
        has_episodes = episode_count > 0
        return {
            "eval_greedy_return": jnp.where(
                has_episodes, completed_mean, jnp.mean(reward_by_env)
            ),
            "eval_greedy_return_std": jnp.where(
                has_episodes,
                jnp.sqrt(completed_variance),
                jnp.std(reward_by_env),
            ),
            "eval_mean_episode_length": jnp.where(
                has_episodes,
                length_sum / safe_count,
                float(args.eval_steps),
            ),
            "eval_episodes": episode_count,
            "eval_partial_return_fallback": jnp.logical_not(has_episodes).astype(
                jnp.float32
            ),
            "eval_reward_per_1000_steps": jnp.sum(reward_by_env)
            / (args.eval_steps * args.eval_num_envs)
            * 1000.0,
        }

    return evaluate


def validate_args(args: Args) -> None:
    if args.num_envs < 1 or args.num_steps < 2:
        raise ValueError("num_envs must be positive and num_steps must be at least 2")
    if args.num_minibatches < 1 or args.num_envs % args.num_minibatches:
        raise ValueError("num_minibatches must be positive and exactly divide num_envs")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.num_iterations < 1:
        raise ValueError("total_timesteps must contain at least one complete rollout")
    if args.update_epochs < 1:
        raise ValueError("update_epochs must be positive")
    if args.hidden_layers < 1:
        raise ValueError("hidden_layers must be positive")
    if args.norm_type.lower().replace("_", "") not in {"layernorm", "none"}:
        raise ValueError("norm_type must be 'layernorm' or 'none'")
    if not 0.0 <= args.q_lambda <= 1.0:
        raise ValueError("q_lambda must be in [0, 1]")
    if args.eval_steps < 1 or args.eval_num_envs < 1:
        raise ValueError("eval_steps and eval_num_envs must be positive")
    if args.optimistic_resets:
        if args.optimistic_reset_ratio < 1:
            raise ValueError("optimistic_reset_ratio must be positive")
        for count in (args.num_envs, args.eval_num_envs):
            if count % min(args.optimistic_reset_ratio, count):
                raise ValueError(
                    "optimistic_reset_ratio must exactly divide each environment count"
                )
    if args.batch_renorm_warmup_steps < 0:
        raise ValueError("batch_renorm_warmup_steps cannot be negative")
    if not 0.0 < args.batch_renorm_momentum <= 1.0:
        raise ValueError("batch_renorm_momentum must be in (0, 1]")
    if not 0.0 < args.jax_mem_fraction <= 1.0:
        raise ValueError("jax_mem_fraction must be in (0, 1]")
    if args.log_interval < 1:
        raise ValueError("log_interval must be positive")
    if args.matrix_diagnostics_interval < 1:
        raise ValueError("matrix_diagnostics_interval must be positive")
    if args.matrix_diagnostics_power_iterations < 1:
        raise ValueError("matrix_diagnostics_power_iterations must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative")


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def _to_python_metrics(metrics: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in jax.device_get(metrics).items()}


def _recent_weighted_mean(
    sums: np.ndarray, counts: np.ndarray, max_count: int = 10_000
) -> Optional[float]:
    selected_sum = 0.0
    selected_count = 0.0
    for value_sum, value_count in zip(reversed(sums), reversed(counts)):
        selected_sum += float(value_sum)
        selected_count += float(value_count)
        if selected_count >= max_count:
            break
    return selected_sum / selected_count if selected_count > 0 else None


def main() -> None:
    args = tyro.cli(Args)
    validate_args(args)
    if args.track and importlib.util.find_spec("wandb") is None:
        raise RuntimeError(
            "W&B tracking was requested but wandb is not installed; install the "
            "Craftax requirements or pass --no-track."
        )
    backend = jax.default_backend()
    requested_cpu = not args.cuda or (args.device or "").lower() == "cpu"
    if requested_cpu and backend != "cpu":
        raise RuntimeError(
            "CPU was requested after JAX initialized a non-CPU backend; pass "
            "--no-cuda or --device cpu directly on the command line."
        )
    if not requested_cpu and backend != "gpu":
        raise RuntimeError(
            "CUDA was requested but JAX did not initialize a GPU backend. Install "
            "a CUDA-enabled JAX build or pass --no-cuda."
        )

    from craftax.craftax_env import make_craftax_env_from_name

    environment = make_craftax_env_from_name(
        args.env_id, auto_reset=not args.optimistic_resets
    )
    env_params = environment.default_params
    observation_shape = tuple(
        int(value) for value in environment.observation_space(env_params).shape
    )
    observation_dim = int(np.prod(observation_shape))
    action_dim = int(environment.action_space(env_params).n)
    train_reset, train_step = make_vector_environment(
        environment,
        env_params,
        num_envs=args.num_envs,
        optimistic_resets=args.optimistic_resets,
        optimistic_reset_ratio=args.optimistic_reset_ratio,
    )
    eval_reset, eval_step = make_vector_environment(
        environment,
        env_params,
        num_envs=args.eval_num_envs,
        optimistic_resets=args.optimistic_resets,
        optimistic_reset_ratio=args.optimistic_reset_ratio,
    )
    network = RNNQNetwork(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        norm_type=args.norm_type,
        norm_input=args.norm_input,
        add_last_action=args.add_last_action,
        batch_renorm_momentum=args.batch_renorm_momentum,
        batch_renorm_eps=args.batch_renorm_eps,
        batch_renorm_max_r=args.batch_renorm_max_r,
        batch_renorm_max_d=args.batch_renorm_max_d,
        batch_renorm_warmup_steps=args.batch_renorm_warmup_steps,
        batch_renorm_smooth=args.batch_renorm_smooth,
    )

    rng = jax.random.PRNGKey(args.seed)
    _, initialization_key, training_key, evaluation_key = jax.random.split(rng, 4)
    variables = network.init(
        initialization_key,
        network.initialize_carry(1),
        jnp.zeros((1, 1, observation_dim), jnp.float32),
        jnp.zeros((1, 1), jnp.bool_),
        jnp.zeros((1, 1), jnp.int32),
        train=False,
    )
    devices = jax.devices()
    ns_steps = args.muon_ns_steps or (5 if backend == "gpu" else 2)
    optimizer = make_optimizer(variables["params"], args, ns_steps=ns_steps)
    initial_train_state = TrainState.create(
        apply_fn=network.apply,
        params=variables["params"],
        batch_stats=variables.get("batch_stats", {}),
        tx=optimizer,
    )

    muon_labels = parameter_labels(
        variables["params"],
        use_muon_input=args.use_muon_input,
        use_muon_output=args.use_muon_output,
    )
    flat_params = traverse_util.flatten_dict(variables["params"])
    flat_labels = traverse_util.flatten_dict(muon_labels)
    parameter_count = int(sum(value.size for value in flat_params.values()))
    muon_parameter_count = int(
        sum(
            flat_params[path].size
            for path, label in flat_labels.items()
            if label == "muon"
        )
        if args.optimizer.lower() == "muon"
        else 0
    )

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
            "framework": "jax_flax",
            "resolved_device": str(devices[0]),
            "resolved_backend": backend,
            "resolved_muon_ns_steps": ns_steps,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "parameter_count": parameter_count,
            "muon_parameter_count": muon_parameter_count,
            "actual_total_timesteps": args.num_iterations * args.batch_size,
            "total_optimizer_steps": (
                args.num_iterations * args.update_epochs * args.num_minibatches
            ),
            "batch_renorm_warmup_fraction": args.batch_renorm_warmup_steps
            / max(
                args.num_iterations * args.update_epochs * args.num_minibatches,
                1,
            ),
        }
    )
    write_json(output_dir / "config.json", config)

    train_function = make_train_function(
        args, network, train_reset, train_step, action_dim
    )
    compile_started = time.time()
    if args.compile_model:
        compiled_train = (
            jax.jit(train_function).lower(initial_train_state, training_key).compile()
        )
    else:
        compiled_train = train_function
    train_compile_seconds = time.time() - compile_started
    print(
        f"compiled training in {train_compile_seconds:.2f}s; "
        f"running {config['actual_total_timesteps']:,} transitions on {devices[0]}"
    )
    training_started = time.time()
    final_train_state, training_metrics = compiled_train(
        initial_train_state, training_key
    )
    final_train_state, training_metrics = jax.block_until_ready(
        (final_train_state, training_metrics)
    )
    training_elapsed = time.time() - training_started
    training_sps = int(config["actual_total_timesteps"] / training_elapsed)

    evaluation_function = make_evaluation_function(args, network, eval_reset, eval_step)
    compile_started = time.time()
    if args.compile_model:
        compiled_evaluation = (
            jax.jit(evaluation_function)
            .lower(
                final_train_state.params,
                final_train_state.batch_stats,
                evaluation_key,
            )
            .compile()
        )
    else:
        compiled_evaluation = evaluation_function
    evaluation_compile_seconds = time.time() - compile_started
    evaluation_started = time.time()
    evaluation_device = compiled_evaluation(
        final_train_state.params,
        final_train_state.batch_stats,
        evaluation_key,
    )
    evaluation_device = jax.block_until_ready(evaluation_device)
    evaluation_elapsed = time.time() - evaluation_started
    evaluation = {
        key: float(value) for key, value in jax.device_get(evaluation_device).items()
    }
    evaluation["eval_episodes"] = int(evaluation["eval_episodes"])
    evaluation["eval_actual_steps"] = args.eval_steps * args.eval_num_envs
    evaluation["eval_sps"] = int(
        evaluation["eval_actual_steps"] / max(evaluation_elapsed, 1.0e-9)
    )

    metrics = {
        key: np.asarray(value)
        for key, value in jax.device_get(training_metrics).items()
    }
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in config.items()),
    )
    progress_fields = [
        "global_step",
        "iteration",
        "learning_rate",
        "epsilon",
        "episodic_return_mean",
        "episodic_length_mean",
        "episode_count",
        "td_loss",
        "q_value_mean",
        "q_value_abs_mean",
        "q_value_abs_max",
        "target_mean",
        "target_abs_mean",
        "target_abs_max",
        "td_error_abs_mean",
        "td_error_abs_max",
        "grad_norm",
        "grad_norm_max",
        "grad_clip_fraction",
        "sps",
    ]
    diagnostic_fields = sorted(
        key for key in metrics if key.startswith("diagnostics/")
    )
    progress_fields.extend(diagnostic_fields)
    wandb_records = []
    progress_path = output_dir / "progress.csv"
    with progress_path.open("w", newline="", encoding="utf-8") as handle:
        progress_writer = csv.DictWriter(handle, fieldnames=progress_fields)
        progress_writer.writeheader()
        for index in range(args.num_iterations):
            iteration = index + 1
            if iteration % args.log_interval and iteration != args.num_iterations:
                continue
            global_step = iteration * args.batch_size
            episode_count = float(metrics["episode_count"][index])
            mean_return = (
                float(metrics["episode_return_sum"][index] / episode_count)
                if episode_count > 0
                else float("nan")
            )
            mean_length = (
                float(metrics["episode_length_sum"][index] / episode_count)
                if episode_count > 0
                else float("nan")
            )
            row = {
                "global_step": global_step,
                "iteration": iteration,
                "learning_rate": float(metrics["learning_rate"][index]),
                "epsilon": float(metrics["epsilon"][index]),
                "episodic_return_mean": mean_return,
                "episodic_length_mean": mean_length,
                "episode_count": int(episode_count),
                "td_loss": float(metrics["td_loss"][index]),
                "q_value_mean": float(metrics["q_value_mean"][index]),
                "q_value_abs_mean": float(metrics["q_value_abs_mean"][index]),
                "q_value_abs_max": float(metrics["q_value_abs_max"][index]),
                "target_mean": float(metrics["target_mean"][index]),
                "target_abs_mean": float(metrics["target_abs_mean"][index]),
                "target_abs_max": float(metrics["target_abs_max"][index]),
                "td_error_abs_mean": float(
                    metrics["td_error_abs_mean"][index]
                ),
                "td_error_abs_max": float(metrics["td_error_abs_max"][index]),
                "grad_norm": float(metrics["grad_norm"][index]),
                "grad_norm_max": float(metrics["grad_norm_max"][index]),
                "grad_clip_fraction": float(
                    metrics["grad_clip_fraction"][index]
                ),
                "sps": training_sps,
            }
            for key in diagnostic_fields:
                row[key] = float(metrics[key][index])
            progress_writer.writerow(row)
            logged = {
                "charts/epsilon": row["epsilon"],
                "charts/learning_rate": row["learning_rate"],
                "charts/SPS": training_sps,
                "losses/td_loss": row["td_loss"],
                "losses/q_values": row["q_value_mean"],
                "losses/q_values_abs_mean": row["q_value_abs_mean"],
                "losses/q_values_abs_max": row["q_value_abs_max"],
                "losses/targets": row["target_mean"],
                "losses/targets_abs_mean": row["target_abs_mean"],
                "losses/targets_abs_max": row["target_abs_max"],
                "losses/td_error_abs_mean": row["td_error_abs_mean"],
                "losses/td_error_abs_max": row["td_error_abs_max"],
                "losses/grad_norm": row["grad_norm"],
                "losses/grad_norm_max": row["grad_norm_max"],
                "losses/grad_clip_fraction": row["grad_clip_fraction"],
            }
            logged.update(
                {
                    key: row[key]
                    for key in diagnostic_fields
                    if np.isfinite(row[key])
                }
            )
            if episode_count > 0:
                logged.update(
                    {
                        "charts/episodic_return_mean": mean_return,
                        "charts/episodic_length_mean": mean_length,
                    }
                )
            for key, value in logged.items():
                writer.add_scalar(key, value, global_step)
            wandb_records.append((global_step, logged))

    training_recent_return = _recent_weighted_mean(
        metrics["episode_return_sum"], metrics["episode_count"]
    )
    training_recent_length = _recent_weighted_mean(
        metrics["episode_length_sum"], metrics["episode_count"]
    )
    total_elapsed = (
        train_compile_seconds
        + training_elapsed
        + evaluation_compile_seconds
        + evaluation_elapsed
    )
    final_diagnostics = {
        key: float(metrics[key][-1])
        for key in diagnostic_fields
        if np.isfinite(float(metrics[key][-1]))
    }
    summary: Dict[str, object] = {
        "status": "ok",
        "algorithm": "discrete_pqn",
        "framework": "jax_flax",
        "environment": "Craftax",
        "env_id": args.env_id,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "global_step": int(config["actual_total_timesteps"]),
        "elapsed_seconds": total_elapsed,
        "training_compile_seconds": train_compile_seconds,
        "evaluation_compile_seconds": evaluation_compile_seconds,
        "training_elapsed_seconds": training_elapsed,
        "evaluation_elapsed_seconds": evaluation_elapsed,
        "sps": training_sps,
        "training_recent_return": training_recent_return,
        "training_recent_length": training_recent_length,
        "final_diagnostics": final_diagnostics,
        "config": config,
        **evaluation,
    }
    if args.save_model:
        model_bytes = serialization.to_bytes(
            {
                "params": final_train_state.params,
                "batch_stats": final_train_state.batch_stats,
            }
        )
        (output_dir / "model.msgpack").write_bytes(model_bytes)

    evaluation_logs = {
        "evaluation/greedy_return": evaluation["eval_greedy_return"],
        "evaluation/greedy_return_std": evaluation["eval_greedy_return_std"],
        "evaluation/reward_per_1000_steps": evaluation["eval_reward_per_1000_steps"],
        "evaluation/episodes": evaluation["eval_episodes"],
        "evaluation/mean_episode_length": evaluation[
            "eval_mean_episode_length"
        ],
        "evaluation/partial_return_fallback": evaluation[
            "eval_partial_return_fallback"
        ],
        "evaluation/elapsed_seconds": evaluation_elapsed,
        "evaluation/SPS": evaluation["eval_sps"],
    }
    for key, value in evaluation_logs.items():
        writer.add_scalar(key, value, int(config["actual_total_timesteps"]))
    writer.close()

    wandb_run = None
    if args.track:
        # Deliberately initialize W&B only after both compiled executables have
        # finished. This keeps networking and W&B bookkeeping entirely outside
        # the measured training/evaluation path while still uploading the full
        # buffered trajectory for every completed seed.
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
        if wandb_records and wandb_records[-1][0] == int(
            config["actual_total_timesteps"]
        ):
            wandb_records[-1][1].update(evaluation_logs)
        else:
            wandb_records.append(
                (int(config["actual_total_timesteps"]), evaluation_logs)
            )
        for global_step, logged in wandb_records:
            wandb_run.log(logged, step=global_step)
        wandb_run.summary.update(
            {
                "eval_greedy_return": evaluation["eval_greedy_return"],
                "eval_reward_per_1000_steps": evaluation["eval_reward_per_1000_steps"],
                "eval_episodes": evaluation["eval_episodes"],
                "eval_greedy_return_std": evaluation["eval_greedy_return_std"],
                "eval_mean_episode_length": evaluation[
                    "eval_mean_episode_length"
                ],
                "sps": training_sps,
                **final_diagnostics,
            }
        )
        summary["wandb_run_id"] = wandb_run.id
        summary["wandb_run_url"] = getattr(wandb_run, "url", None)
        wandb_run.finish()
        summary["wandb_uploaded"] = True
    else:
        summary["wandb_uploaded"] = False
    write_json(output_dir / "summary.json", summary)
    print(
        f"completed step={config['actual_total_timesteps']:,} "
        f"optimizer={args.optimizer} SPS={training_sps:,} "
        f"eval_return={evaluation['eval_greedy_return']:.4f}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

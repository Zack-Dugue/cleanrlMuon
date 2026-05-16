# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import csv
import json
import os
import random
import shutil
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RacingProject"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to record one deterministic final-eval video at the end"""

    anneal_entropy: bool = True
    """anneal dat entropy"""
    # Algorithm specific arguments
    env_id: str = "CarRacing-v3"
    """the id of the environment"""
    total_timesteps: int = 1000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    aux_learning_rate: float = 2.5e-4
    """Aux LR"""

    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.03
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    save_path: str = "agent.pth"
    """final path; after training this will contain the best eval checkpoint if eval ran"""
    best_save_path: str | None = None
    """optional rolling-best checkpoint path; default is save_path with .best before the extension"""
    eval_interval_timesteps: int = 1_000_000
    """run deterministic eval about this often; <=0 disables eval checkpointing"""
    eval_episodes: int = 8
    """number of full episodes/tracks for each intermittent eval"""
    eval_num_envs: int = 32
    """number of parallel envs used for intermittent eval; keep this reasonably large so eval is batched"""
    eval_seed_offset: int = 10_000
    """eval env seed is seed + eval_seed_offset + global_step"""
    final_eval_episodes: int = 300
    """number of full tracks for the final deterministic eval"""
    final_eval_num_envs: int = 64
    """number of parallel envs used for final deterministic eval; use a decent batch so 300 tracks does not crawl"""
    final_eval_results_path: str = "checkpoints/carracing_final_eval_results.json"
    """where to write final 300-track eval mean/std/min/max and all returns"""
    final_video_path: str = "videos/carracing_final_eval.mp4"
    """where to write one deterministic final-eval rollout video"""
    final_video_fps: int = 30
    """frames per second for the final-eval video"""
    final_video_max_steps: int = 1001
    """maximum number of environment steps to record in the final-eval video"""
    final_video_seed_offset: int = 999_999
    """video env seed is seed + eval_seed_offset + final_video_seed_offset"""
    action_history_len: int = 4
    """number of previous continuous actions to concatenate with visual features"""
    actor_log_std_init: float = 0
    """initial value for learned log std; exp(-1) ~= 0.37"""
    std_lr_mult: float = 10.0
    """learning-rate multiplier for actor_log_std parameter group"""

    debug_optimizer_logging: bool = True
    """log extra optimizer/policy diagnostics for Adam vs Muon-style comparisons"""
    debug_light_interval: int = 1
    """log cheap policy/loss/action diagnostics every N PPO iterations"""
    debug_delta_interval: int = 50
    """log expensive parameter-delta diagnostics every N PPO iterations; <=0 disables"""
    debug_policy_sample_size: int = 2048
    """number of flattened rollout samples used for policy sharpness diagnostics"""
    debug_tracked_param_names: str = "conv1.weight,conv2.weight,conv3.weight,trunk_fc.weight,actor_mean.weight,critic_out.weight,actor_log_std"
    """comma-separated parameter names for delta/grad/param norm diagnostics"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

class CarRacingFrameStack(gym.Wrapper):
    """
    Frame-stack wrapper for vectorized EnvPool CarRacing.

    Input observation:
        [num_envs, 96, 96, 3]

    Output observation:
        [num_envs, 96, 96, 3 * stack_size]

    This keeps the HWC layout used by EnvPool/Gymnasium, then your model's
    _normalize_input() still converts HWC -> CHW with torch.permute.
    """

    def __init__(self, env, stack_size=4):
        super().__init__(env)

        self.num_envs = int(getattr(env, "num_envs", 1))
        self.stack_size = int(stack_size)

        base_obs_space = getattr(env, "single_observation_space", env.observation_space)
        base_action_space = getattr(env, "single_action_space", env.action_space)

        if len(base_obs_space.shape) != 3:
            raise ValueError(
                f"Expected image observation shape [H, W, C], got {base_obs_space.shape}"
            )

        h, w, c = base_obs_space.shape

        self.single_observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, c * self.stack_size),
            dtype=base_obs_space.dtype,
        )
        self.observation_space = self.single_observation_space

        self.single_action_space = base_action_space
        self.action_space = base_action_space

        self.frames = deque(maxlen=self.stack_size)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return_info = True
        else:
            obs = out
            info = {}
            return_info = False

        obs = np.asarray(obs)

        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(obs.copy())

        stacked_obs = np.concatenate(list(self.frames), axis=-1)

        if return_info:
            return stacked_obs, info
        return stacked_obs

    def step(self, action):
        out = self.env.step(action)

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            self.frames.append(np.asarray(obs).copy())
            stacked_obs = np.concatenate(list(self.frames), axis=-1)
            return stacked_obs, reward, terminated, truncated, info

        if len(out) == 4:
            obs, reward, done, info = out
            self.frames.append(np.asarray(obs).copy())
            stacked_obs = np.concatenate(list(self.frames), axis=-1)
            return stacked_obs, reward, done, info

        raise RuntimeError(f"Expected env.step() to return 4 or 5 values, got {len(out)}")

class RecordEpisodeStatistics(gym.Wrapper):
    """
    Episode return/length tracker that works with EnvPool-style vector envs
    and Gymnasium-style APIs.

    It supports either:

        reset() -> obs
        step()  -> obs, reward, done, info

    or:

        reset() -> obs, info
        step()  -> obs, reward, terminated, truncated, info

    It returns the same API style it receives from step():
        - If env.step gives 4 values, this wrapper returns 4 values.
        - If env.step gives 5 values, this wrapper returns 5 values.

    It injects:
        info["r"] = episode returns before reset masking
        info["l"] = episode lengths before reset masking
        info["terminated"] = done/terminated array
        info["truncated"] = truncated array if available, else zeros
        info["done"] = final done array
    """

    def __init__(self, env, deque_size=100):
        super().__init__(env)

        self.num_envs = int(getattr(env, "num_envs", 1))

        # CleanRL expects these names. EnvPool may expose read-only properties,
        # so we define them on the wrapper instead of mutating the raw env.
        self.single_observation_space = getattr(
            env,
            "single_observation_space",
            env.observation_space,
        )
        self.single_action_space = getattr(
            env,
            "single_action_space",
            env.action_space,
        )

        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        # Gymnasium reset returns (obs, info); old Gym/EnvPool-gym often returns obs.
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return_info = True
        else:
            obs = out
            info = {}
            return_info = False

        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

        if return_info:
            return obs, info
        return obs

    def step(self, action):
        out = self.env.step(action)

        if not isinstance(out, tuple):
            raise RuntimeError(f"Expected env.step(action) to return a tuple, got {type(out)}")

        # Gymnasium API: obs, reward, terminated, truncated, info
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            done = np.logical_or(terminated, truncated)
            return_gymnasium_api = True

        # Old Gym / EnvPool gym API: obs, reward, done, info
        elif len(out) == 4:
            obs, reward, done, info = out
            done = np.asarray(done, dtype=bool)

            # EnvPool sometimes includes these in info; otherwise infer them.
            if isinstance(info, dict) and "terminated" in info:
                terminated = np.asarray(info["terminated"], dtype=bool)
            else:
                terminated = done

            if isinstance(info, dict) and "truncated" in info:
                truncated = np.asarray(info["truncated"], dtype=bool)
            else:
                truncated = np.zeros_like(done, dtype=bool)

            return_gymnasium_api = False

        else:
            raise RuntimeError(
                f"Expected env.step(action) to return 4 or 5 values, got {len(out)}"
            )

        reward = np.asarray(reward, dtype=np.float32)

        # EnvPool info should be a dict of arrays. If not, make a mutable dict.
        if info is None:
            info = {}
        elif not isinstance(info, dict):
            info = dict(info)

        self.episode_returns += reward
        self.episode_lengths += 1

        # Store the just-finished episode stats before zeroing done envs.
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths

        info["r"] = self.returned_episode_returns.copy()
        info["l"] = self.returned_episode_lengths.copy()
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["done"] = done

        # Reset counters for envs whose episode ended.
        self.episode_returns[done] = 0.0
        self.episode_lengths[done] = 0

        if return_gymnasium_api:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ConvSimpleAgent(nn.Module):
    """
    PQN-style ConvNet actor-critic for Atari EnvPool observations,
    with input BatchRenorm2d.

    Expected input:
      x shape [batch, 4, 84, 84], uint8-ish pixels in [0, 255]

    Input normalization:
      x.float() / 255.0
      BatchRenorm2d(C)

    For Atari frame stacks:
      C = 4

    So BatchRenorm2d tracks one running distribution per stacked-frame channel,
    across batch and spatial dimensions. This is much more natural for convnets
    than flattening the whole image and using BatchRenorm1d(obs_dim).

    PQN-style encoder:
      Conv2d -> LayerNorm -> ReLU
      Conv2d -> LayerNorm -> ReLU
      Conv2d -> LayerNorm -> ReLU
      Flatten
      Linear -> LayerNorm -> ReLU

    PPO-style separate heads:
      actor_out:  Linear(hidden_dim, action_dim), std=0.01
      critic_out: Linear(hidden_dim, 1), std=1.0

    Muon routing:
      default:
        Muon gets trunk_fc.weight

      use_muon_input=True:
        also sends conv1.weight, conv2.weight, conv3.weight to Muon

      use_muon_output=True:
        also sends actor_out.weight and critic_out.weight to Muon

      Adam gets:
        BatchRenorm params,
        LayerNorm params,
        all biases,
        and anything not explicitly routed to Muon.
    """

    def __init__(
        self,
        envs,
        hidden_dim=1024,
        *,
        use_muon_input=False,
        use_muon_output=False,
        continuous_eps=.00001,
        action_history_len=4,
        actor_log_std_init=-1.0,
    ):
        super().__init__()

        self.use_muon_input = use_muon_input
        self.use_muon_output = use_muon_output

        obs_shape = envs.single_observation_space.shape
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim
        self.action_history_len = int(action_history_len)
        self.action_history_dim = self.action_history_len * self.action_dim
        model_input_dim = hidden_dim + self.action_history_dim
        self.register_buffer( "action_low",torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer( "action_high",torch.tensor(envs.single_action_space.high, dtype=torch.float32))
        if len(obs_shape) != 3:
            raise ValueError(
                f"Expected Atari image observation shape [C,H,W], got {obs_shape}"
            )

        h, w, c = obs_shape

        if c != 4:
            print(
                f"[BetterSimpleAgent/PQN-BRN2d warning] expected 4 stacked frames, got C={c}. "
                "Continuing anyway."
            )

        if h != 84 or w != 84:
            print(
                f"[BetterSimpleAgent/PQN-BRN2d warning] expected 84x84 Atari obs, got H={h}, W={w}. "
                "LayerNorm shapes assume standard Atari preprocessing."
            )

        if hidden_dim != 512:
            print(
                f"[BetterSimpleAgent/PQN-BRN2d warning] PQN usually uses hidden_dim=512. "
                f"You passed hidden_dim={hidden_dim}."
            )

        # Input BatchRenorm over image channels.
        #
        # For x shape [B, C, H, W], BatchRenorm2d(C) normalizes each channel
        # using statistics computed over B,H,W.
        #
        # For Atari stacked grayscale frames, C=4, so each frame index gets its
        # own running statistics.
        # self.input_brn = BatchRenorm2d(
        #     c,
        #     eps=brn_eps,
        #     momentum=brn_momentum,
        #     max_r=brn_max_r,
        #     max_d=brn_max_d,
        #     warmup_steps=brn_warmup_steps,
        #     smooth=brn_smooth,
        # )

        # ----- PQN-style conv encoder -----
        self.conv1 = layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4))
        self.ln1 = nn.LayerNorm([32, 23, 23])

        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.ln2 = nn.LayerNorm([64, 10, 10])

        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.ln3 = nn.LayerNorm([64, 8, 8])

        self.flatten = nn.Flatten()

        # For 84x84 Atari:
        # 84 -> conv1 -> 20
        # 20 -> conv2 -> 9
        # 9  -> conv3 -> 7
        # so final conv output is 64 * 7 * 7 = 3136.
        self.trunk_fc = layer_init(nn.Linear(4096, hidden_dim))
        self.trunk_ln = nn.LayerNorm(hidden_dim)

        self.act = nn.GELU()

        # ----- Separate PPO actor/value heads -----
        # self.actor_fc = layer_init(nn.Linear(hidden_dim, hidden_dim))
        # self.critic_fc = layer_init(nn.Linear(hidden_dim, hidden_dim))
        # self.actor_ln = nn.LayerNorm(hidden_dim)
        # self.critic_ln = nn.LayerNorm(hidden_dim)

        self.continuous_eps = continuous_eps

        self.actor_mean = layer_init(
            nn.Linear(model_input_dim, action_dim),
            std=0.01,
        )

        self.actor_mean.bias.data = torch.Tensor([0, 0, -2])

        # Learned state-independent log standard deviation.
        # This is much more stable for PPO than predicting log_std with a second head.
        self.actor_log_std = nn.Parameter(torch.full((1, action_dim), float(actor_log_std_init)))

        self.critic_out = layer_init(
            nn.Linear(model_input_dim, 1),
            std=1.0,
        )

        total_params = sum(p.numel() for p in self.parameters())
        muon_params, adam_params = self.get_split_params()

        print(
            f"[BetterSimpleAgent/PQN-BRN2d] obs_shape={obs_shape}, "
            f"hidden_dim={hidden_dim}, action_dim={action_dim}, "
            f"action_history_len={self.action_history_len}, model_input_dim={model_input_dim}"
        )
        print("[BetterSimpleAgent/PQN-BRN2d] input normalization: BatchRenorm2d(C)")
        print(f"[BetterSimpleAgent/PQN-BRN2d] total parameters: {total_params:,}")
        print(f"[BetterSimpleAgent/PQN-BRN2d] Muon parameters: {sum(p.numel() for p in muon_params):,}")
        print(f"[BetterSimpleAgent/PQN-BRN2d] Adam parameters: {sum(p.numel() for p in adam_params):,}")
        print(
            f"[BetterSimpleAgent/PQN-BRN2d] "
            f"use_muon_input={use_muon_input}, use_muon_output={use_muon_output}"
        )

    def _normalize_input(self, x):
        """
        Normalize raw Atari image input.

        Input:
          x: [B, C, H, W], usually uint8 in [0, 255]

        Output:
          x: [B, C, H, W], float normalized by /255 and BatchRenorm2d.
        """
        x = x.float() / 255.0
        x = torch.permute(x, (0, 3, 1, 2))
        # x = self.input_brn(x)
        return x

    def _features(self, x):
        x = self._normalize_input(x)

        x = self.conv1(x)
        x = self.act(x)
        x = self.ln1(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.ln2(x)

        x = self.conv3(x)
        x = self.act(x)
        x = self.ln3(x)


        x = self.flatten(x)

        x = self.trunk_fc(x)
        x = self.act(x)
        # x = self.trunk_ln(x)

        return x

    def _features_with_action_history(self, x, action_history=None):
        """
        Build the model input from visual features plus the previous-action stack.

        action_history shape: [B, action_history_len, action_dim]. It stores the
        actual continuous env actions from previous frames, aligned with the
        stacked visual frames. If omitted, zeros are used.
        """
        features = self._features(x)

        if self.action_history_len <= 0:
            return features

        batch_size = features.shape[0]
        if action_history is None:
            action_history = torch.zeros(
                batch_size,
                self.action_history_len,
                self.action_dim,
                dtype=features.dtype,
                device=features.device,
            )
        else:
            action_history = action_history.to(device=features.device, dtype=features.dtype)
            action_history = action_history.reshape(batch_size, self.action_history_dim)
            return torch.cat([features, action_history], dim=1)

        action_history = action_history.reshape(batch_size, self.action_history_dim)
        return torch.cat([features, action_history], dim=1)

    def get_split_params(self):
        """
        Returns:
            muon_params, adam_params

        Default:
          Muon:
            trunk_fc.weight

          Adam:
            input_brn params
            conv weights unless use_muon_input=True
            output heads unless use_muon_output=True
            all biases
            all LayerNorm params

        Optional:
          use_muon_input=True:
            conv1.weight
            conv2.weight
            conv3.weight

          use_muon_output=True:
            actor_out.weight
            critic_out.weight
        """

        muon_params = [
            # self.actor_fc.weight,
            # self.critic_fc.weight,
            self.trunk_fc.weight,
        ]

        if self.use_muon_input:
            muon_params.extend([
                # self.conv1.weight,
                self.conv2.weight,
                self.conv3.weight,
            ])

        if self.use_muon_output:
            muon_params.extend([
                self.actor_mean.weight,
                self.critic_out.weight,
            ])

        muon_ids = {id(p) for p in muon_params}

        adam_params = [
            p for p in self.parameters()
            if id(p) not in muon_ids
        ]

        return muon_params, adam_params

    def get_value(self, x, action_history=None):
        features = self._features_with_action_history(x, action_history)
        # return self.critic_out(self.act(self.critic_ln(self.critic_fc(features))))
        return self.critic_out(features)

    def get_action_and_value(self, x, action_history=None, action=None, deterministic: bool = False):
        features = self._features_with_action_history(x, action_history)

        # actor_features = self.act(self.actor_ln(self.actor_fc(features)))
        actor_features = features
        actor_mean = self.actor_mean(actor_features)

        actor_log_std = self.actor_log_std.expand_as(actor_mean)
        actor_log_std = torch.clamp(actor_log_std, -5.0, 2.0)
        actor_std = torch.exp(actor_log_std)

        normal = Normal(actor_mean, actor_std)

        if action is None:
            if deterministic:
                # Deterministic mode for evaluation/checkpoint selection.
                # The mean lives in unconstrained pre-sigmoid action space.
                raw_action = actor_mean
            else:
                raw_action = normal.rsample()
            squashed_action = torch.sigmoid(raw_action)
            squashed_action = squashed_action.clamp(
                self.continuous_eps,
                1.0 - self.continuous_eps,
            )
            env_action = squashed_action * (self.action_high - self.action_low) + self.action_low
        else:
            env_action = action

            squashed_action = (env_action - self.action_low) / (
                    self.action_high - self.action_low
            )
            squashed_action = squashed_action.clamp(
                self.continuous_eps,
                1.0 - self.continuous_eps,
            )

            raw_action = torch.logit(squashed_action, eps=self.continuous_eps)

        log_prob = normal.log_prob(raw_action)

        log_prob -= torch.log(
            squashed_action * (1.0 - squashed_action) + self.continuous_eps
        )

        log_prob -= torch.log(
            self.action_high - self.action_low
        )

        log_prob = log_prob.sum(dim=-1)

        # Better entropy proxy for the transformed action.
        entropy = normal.entropy()
        entropy += torch.log(
            squashed_action * (1.0 - squashed_action) + self.continuous_eps
        )
        entropy += torch.log(
            self.action_high - self.action_low
        )
        entropy = entropy.sum(dim=-1)

        # value = self.critic_out(
        #     self.act(self.critic_ln(self.critic_fc(features)))
        # )
        value = self.critic_out(features)

        return env_action, log_prob, entropy, value


def default_best_save_path(save_path: str) -> str:
    """Return a path like foo.best.pt for rolling best-eval checkpoints."""
    root, ext = os.path.splitext(save_path)
    if ext:
        return f"{root}.best{ext}"
    return f"{save_path}.best.pt"


def make_carracing_envs(args: Args, *, num_envs: int, seed: int):
    """Build the same wrapped CarRacing env stack for train/eval."""
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=num_envs,
        seed=seed,
    )
    envs = CarRacingFrameStack(envs, stack_size=4)
    envs = RecordEpisodeStatistics(envs)
    return envs


def save_checkpoint(
    path: str,
    agent: nn.Module,
    optimizer: optim.Optimizer,
    args: Args,
    global_step: int,
    run_name: str,
    *,
    best_eval_return: float | None = None,
    eval_returns: list[float] | None = None,
    is_best: bool = False,
    final_eval_stats: dict | None = None,
):
    """Save the agent plus eval metadata."""
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "global_step": global_step,
            "run_name": run_name,
            "best_eval_return": best_eval_return,
            "eval_returns": eval_returns,
            "is_best": is_best,
            "final_eval_stats": final_eval_stats,
        },
        path,
    )


def write_final_eval_results(path: str, final_eval_stats: dict, args: Args, run_name: str, global_step: int):
    """Write final evaluation results to JSON and a companion CSV of per-track returns."""
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    payload = {
        "run_name": run_name,
        "global_step": int(global_step),
        "env_id": args.env_id,
        "seed": int(args.seed),
        "num_episodes": int(final_eval_stats["num_episodes"]),
        "num_envs": int(final_eval_stats["num_envs"]),
        "eval_seed": int(final_eval_stats["eval_seed"]),
        "checkpoint_path": final_eval_stats.get("checkpoint_path"),
        "mean_return": float(final_eval_stats["mean_return"]),
        "std_return": float(final_eval_stats["std_return"]),
        "min_return": float(final_eval_stats["min_return"]),
        "max_return": float(final_eval_stats["max_return"]),
        "returns": [float(x) for x in final_eval_stats["returns"]],
    }

    # Optional video metadata, if a final rollout video was recorded.
    for extra_key in ["video_path", "video_return", "video_length", "video_seed"]:
        if extra_key in final_eval_stats:
            value = final_eval_stats[extra_key]
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            payload[extra_key] = value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    root, _ = os.path.splitext(path)
    csv_path = f"{root}.returns.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_index", "return"])
        for i, ret in enumerate(payload["returns"]):
            writer.writerow([i, ret])

    return os.path.abspath(path), os.path.abspath(csv_path)


def _safe_tb_name(name: str) -> str:
    """Make a parameter/module name friendly for TensorBoard/W&B scalar paths."""
    return name.replace(".", "_").replace("/", "_")


def _parse_tracked_param_names(names: str) -> set[str]:
    return {x.strip() for x in str(names).split(",") if x.strip()}


@torch.no_grad()
def log_action_behavior_debug(writer: SummaryWriter, actions: torch.Tensor, global_step: int):
    """
    Continuous-action behavior diagnostics over the just-collected rollout.

    actions shape:
      [num_steps, num_envs, action_dim]

    These are useful for seeing whether one optimizer learns smoother/more
    persistent control behavior than another.
    """
    flat = actions.detach().float().reshape(-1, actions.shape[-1])

    writer.add_scalar("debug_behavior/action_mean", flat.mean().item(), global_step)
    writer.add_scalar("debug_behavior/action_std", flat.std().item(), global_step)
    writer.add_scalar("debug_behavior/action_abs_mean", flat.abs().mean().item(), global_step)

    for dim in range(flat.shape[-1]):
        writer.add_scalar(f"debug_behavior/action_dim_{dim}_mean", flat[:, dim].mean().item(), global_step)
        writer.add_scalar(f"debug_behavior/action_dim_{dim}_std", flat[:, dim].std().item(), global_step)

    if actions.shape[0] > 1:
        diffs = actions[1:].detach().float() - actions[:-1].detach().float()
        writer.add_scalar("debug_behavior/action_repeat_l1", diffs.abs().mean().item(), global_step)
        writer.add_scalar("debug_behavior/action_repeat_l2", torch.sqrt((diffs.pow(2).mean()) + 1e-12).item(), global_step)
        writer.add_scalar("debug_behavior/action_switch_proxy", diffs.abs().sum(dim=-1).mean().item(), global_step)


@torch.no_grad()
def log_policy_sharpness_debug(
    writer: SummaryWriter,
    agent: ConvSimpleAgent,
    b_obs: torch.Tensor,
    b_action_histories: torch.Tensor,
    global_step: int,
    *,
    sample_size: int,
):
    """
    Continuous-policy analog of logit/probability sharpness diagnostics.

    For CarRacing, the policy is a Normal in raw pre-sigmoid action space plus
    sigmoid/action-range correction. So instead of max_action_prob/logit_std,
    we log actor mean/std/log_std and the squashed mean action distribution.
    """
    n = min(int(sample_size), b_obs.shape[0])
    if n <= 0:
        return

    obs_sample = b_obs[:n]
    hist_sample = b_action_histories[:n]

    features = agent._features_with_action_history(obs_sample, hist_sample)
    actor_mean = agent.actor_mean(features)
    actor_log_std = torch.clamp(agent.actor_log_std.expand_as(actor_mean), -5.0, 2.0)
    actor_std = torch.exp(actor_log_std)
    squashed_mean = torch.sigmoid(actor_mean)
    env_mean = squashed_mean * (agent.action_high - agent.action_low) + agent.action_low

    writer.add_scalar("debug_policy/actor_mean_abs", actor_mean.abs().mean().item(), global_step)
    writer.add_scalar("debug_policy/actor_mean_std", actor_mean.std().item(), global_step)
    writer.add_scalar("debug_policy/actor_mean_norm", actor_mean.norm(dim=-1).mean().item(), global_step)

    writer.add_scalar("debug_policy/actor_log_std_mean", actor_log_std.mean().item(), global_step)
    writer.add_scalar("debug_policy/actor_log_std_min", actor_log_std.min().item(), global_step)
    writer.add_scalar("debug_policy/actor_log_std_max", actor_log_std.max().item(), global_step)

    writer.add_scalar("debug_policy/actor_std_mean", actor_std.mean().item(), global_step)
    writer.add_scalar("debug_policy/actor_std_min", actor_std.min().item(), global_step)
    writer.add_scalar("debug_policy/actor_std_max", actor_std.max().item(), global_step)

    writer.add_scalar("debug_policy/squashed_mean_mean", squashed_mean.mean().item(), global_step)
    writer.add_scalar("debug_policy/squashed_mean_std", squashed_mean.std().item(), global_step)
    writer.add_scalar("debug_policy/squashed_mean_min", squashed_mean.min().item(), global_step)
    writer.add_scalar("debug_policy/squashed_mean_max", squashed_mean.max().item(), global_step)

    writer.add_scalar("debug_policy/env_mean_action_mean", env_mean.mean().item(), global_step)
    writer.add_scalar("debug_policy/env_mean_action_std", env_mean.std().item(), global_step)

    # Entropy of the raw Normal before sigmoid correction. This is not the same
    # as the corrected entropy proxy used in the PPO loss, but it cleanly shows
    # whether the learned Gaussian std is shrinking or growing.
    raw_normal_entropy = 0.5 + 0.5 * np.log(2.0 * np.pi) + actor_log_std
    writer.add_scalar("debug_policy/raw_normal_entropy_mean", raw_normal_entropy.sum(dim=-1).mean().item(), global_step)


def snapshot_tracked_params(agent: nn.Module, tracked_names: set[str]) -> dict[str, torch.Tensor]:
    """
    Clone a tiny selected subset of params before a PPO update so we can measure
    actual optimizer movement after the full PPO update.
    """
    out = {}
    for name, p in agent.named_parameters():
        if name in tracked_names:
            out[name] = p.detach().clone()
    return out


@torch.no_grad()
def log_param_delta_debug(
    writer: SummaryWriter,
    agent: nn.Module,
    tracked_before: dict[str, torch.Tensor],
    global_step: int,
):
    """
    Log actual parameter deltas over a full PPO iteration.

    This is the key diagnostic for checking whether the same nominal LR causes
    Adam and Muon to move trunk/conv/head params by similar relative amounts.
    """
    if not tracked_before:
        return

    for name, p in agent.named_parameters():
        if name not in tracked_before:
            continue

        before = tracked_before[name]
        cur = p.detach()
        delta = cur - before
        grad = p.grad.detach() if p.grad is not None else None

        safe = _safe_tb_name(name)
        param_norm = cur.norm().item()
        delta_norm = delta.norm().item()
        grad_norm = 0.0 if grad is None else grad.norm().item()

        writer.add_scalar(f"debug_update/{safe}/param_norm", param_norm, global_step)
        writer.add_scalar(f"debug_update/{safe}/delta_norm", delta_norm, global_step)
        writer.add_scalar(f"debug_update/{safe}/grad_norm", grad_norm, global_step)
        writer.add_scalar(f"debug_update/{safe}/delta_over_param", delta_norm / (param_norm + 1e-12), global_step)
        writer.add_scalar(f"debug_update/{safe}/delta_over_grad", delta_norm / (grad_norm + 1e-12), global_step)

        if grad is not None:
            writer.add_scalar(f"debug_grad/{safe}/abs_mean", grad.abs().mean().item(), global_step)
            writer.add_scalar(f"debug_grad/{safe}/std", grad.std().item(), global_step)


@torch.no_grad()
def log_layernorm_debug(writer: SummaryWriter, agent: nn.Module, global_step: int):
    """Lightweight check for LayerNorm compensation/drift."""
    for module_name, m in agent.named_modules():
        if isinstance(m, nn.LayerNorm):
            safe = _safe_tb_name(module_name)
            writer.add_scalar(f"debug_norm/{safe}/weight_mean", m.weight.mean().item(), global_step)
            writer.add_scalar(f"debug_norm/{safe}/weight_std", m.weight.std().item(), global_step)
            writer.add_scalar(f"debug_norm/{safe}/bias_mean", m.bias.mean().item(), global_step)
            writer.add_scalar(f"debug_norm/{safe}/bias_std", m.bias.std().item(), global_step)


@torch.no_grad()
def evaluate_deterministic_policy(
    agent: ConvSimpleAgent,
    args: Args,
    device: torch.device,
    *,
    num_episodes: int,
    num_envs: int,
    seed: int,
):
    """
    Evaluate with deterministic actions: action = sigmoid(actor_mean), no sampling.

    Maintains the same previous-action history as training, because that history
    is part of the model input.
    """
    if num_episodes <= 0:
        return float("nan"), float("nan"), []

    eval_num_envs = max(1, min(int(num_envs), int(num_episodes)))
    eval_envs = make_carracing_envs(args, num_envs=eval_num_envs, seed=seed)

    was_training = agent.training
    agent.eval()

    reset_out = eval_envs.reset()
    if isinstance(reset_out, tuple):
        obs_np, _ = reset_out
    else:
        obs_np = reset_out

    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    done = torch.zeros(eval_num_envs, dtype=torch.float32, device=device)
    action_history = torch.zeros(
        (eval_num_envs, args.action_history_len) + eval_envs.single_action_space.shape,
        dtype=torch.float32,
        device=device,
    )

    returns: list[float] = []
    lengths: list[int] = []

    while len(returns) < num_episodes:
        action, _, _, _ = agent.get_action_and_value(
            obs,
            action_history,
            action=None,
            deterministic=True,
        )

        step_out = eval_envs.step(action.cpu().numpy())
        if len(step_out) == 5:
            next_obs_np, reward, terminated, truncated, info = step_out
            done_np = np.logical_or(terminated, truncated)
        elif len(step_out) == 4:
            next_obs_np, reward, done_np, info = step_out
        else:
            raise RuntimeError(f"Expected env.step() to return 4 or 5 values, got {len(step_out)}")

        obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
        done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

        if args.action_history_len > 0:
            action_history = torch.roll(action_history, shifts=-1, dims=1)
            action_history[:, -1] = action.detach()
            done_mask = done.bool()
            if done_mask.any():
                action_history[done_mask] = 0.0

        for idx, done_flag in enumerate(done_np):
            if done_flag:
                returns.append(float(info["r"][idx]))
                lengths.append(int(info["l"][idx]))
                if len(returns) >= num_episodes:
                    break

    eval_envs.close()
    if was_training:
        agent.train()

    returns = returns[:num_episodes]
    lengths = lengths[:num_episodes]
    return float(np.mean(returns)), float(np.std(returns)), returns


@torch.no_grad()
def record_deterministic_video(
    agent: ConvSimpleAgent,
    args: Args,
    device: torch.device,
    *,
    video_path: str,
    seed: int,
    fps: int,
    max_steps: int,
):
    """
    Record one deterministic single-environment rollout to an MP4.

    This intentionally uses gymnasium.make(..., render_mode="rgb_array") instead
    of EnvPool, because EnvPool is great for batched training/eval but awkward for
    reliably extracting RGB frames. The observation preprocessing is kept aligned
    with training by manually doing the same 4-frame HWC stack used by
    CarRacingFrameStack.
    """
    if imageio is None:
        raise RuntimeError(
            "imageio is not installed/importable. Install it with: "
            "pip install imageio imageio-ffmpeg"
        )

    save_dir = os.path.dirname(video_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    env = gym.make(args.env_id, render_mode="rgb_array")

    was_training = agent.training
    agent.eval()

    reset_out = env.reset(seed=seed)
    if isinstance(reset_out, tuple):
        obs_np, _ = reset_out
    else:
        obs_np = reset_out

    frame_stack = deque(maxlen=4)
    obs_np = np.asarray(obs_np)
    for _ in range(4):
        frame_stack.append(obs_np.copy())

    action_history = torch.zeros(
        (1, args.action_history_len) + env.action_space.shape,
        dtype=torch.float32,
        device=device,
    )

    frames = []
    initial_frame = env.render()
    if initial_frame is not None:
        frames.append(np.asarray(initial_frame))

    total_return = 0.0
    length = 0
    terminated = False
    truncated = False

    while not (terminated or truncated) and length < int(max_steps):
        stacked_obs = np.concatenate(list(frame_stack), axis=-1)
        obs = torch.as_tensor(stacked_obs[None, ...], dtype=torch.float32, device=device)

        action, _, _, _ = agent.get_action_and_value(
            obs,
            action_history,
            action=None,
            deterministic=True,
        )
        action_np = action[0].detach().cpu().numpy()

        step_out = env.step(action_np)
        if len(step_out) == 5:
            next_obs_np, reward, terminated, truncated, info = step_out
        elif len(step_out) == 4:
            next_obs_np, reward, done, info = step_out
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError(f"Expected env.step() to return 4 or 5 values, got {len(step_out)}")

        total_return += float(reward)
        length += 1

        rendered = env.render()
        if rendered is not None:
            frames.append(np.asarray(rendered))

        frame_stack.append(np.asarray(next_obs_np).copy())

        if args.action_history_len > 0:
            action_history = torch.roll(action_history, shifts=-1, dims=1)
            action_history[:, -1] = action.detach()

    env.close()

    if was_training:
        agent.train()

    if not frames:
        raise RuntimeError("No frames were captured from the render environment.")

    imageio.mimsave(video_path, frames, fps=int(fps))

    return os.path.abspath(video_path), float(total_return), int(length)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,

        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(f"the device we're using is: {device}")

    # env setup
    envs = make_carracing_envs(args, num_envs=args.num_envs, seed=args.seed)
    # assert isinstance(envs.action_space, gym.spaces.Continuous), "only continuous action space is supported"

    agent = ConvSimpleAgent(
        envs,
        action_history_len=args.action_history_len,
        actor_log_std_init=args.actor_log_std_init,
    ).to(device)
    std_params = []
    main_params = []

    for name, param in agent.named_parameters():
        if name == "actor_log_std":
            std_params.append(param)
        else:
            main_params.append(param)

    optimizer = optim.AdamW(
        [
            {
                "params": main_params,
                "lr": args.learning_rate,
                "weight_decay": 1e-5,
            },
            {
                "params": std_params,
                "lr": args.learning_rate * args.std_lr_mult,
                "weight_decay": 0.0,
            },
        ],
        eps=1e-5,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    action_histories = torch.zeros(
        (args.num_steps, args.num_envs, args.action_history_len) + envs.single_action_space.shape,
        device=device,
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    reset_out = envs.reset()
    if isinstance(reset_out, tuple):
        next_obs, reset_info = reset_out
    else:
        next_obs = reset_out
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    next_action_history = torch.zeros(
        (args.num_envs, args.action_history_len) + envs.single_action_space.shape,
        dtype=torch.float32,
        device=device,
    )

    best_save_path = args.best_save_path or default_best_save_path(args.save_path)
    best_eval_return = -float("inf")
    best_eval_returns: list[float] | None = None
    last_eval_step = 0

    tracked_param_names = _parse_tracked_param_names(args.debug_tracked_param_names)
    missing_tracked = sorted([name for name in tracked_param_names if name not in dict(agent.named_parameters())])
    if args.debug_optimizer_logging:
        print(f"[debug] tracked_param_names={sorted(tracked_param_names)}")
        if missing_tracked:
            print(f"[debug warning] requested tracked params not found: {missing_tracked}")

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lr_frac = max(0.1, frac)
            for group in optimizer.param_groups:
                group["lr"] = lr_frac * group["initial_lr"]

        if args.anneal_entropy:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            ent_coef = frac**(2) * args.ent_coef
        else:
            ent_coef = args.ent_coef

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            action_histories[step] = next_action_history

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, next_action_history)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            step_out = envs.step(action.cpu().numpy())

            # Support both Gymnasium-style 5-tuples and old Gym/EnvPool-style 4-tuples.
            if len(step_out) == 5:
                next_obs, reward, next_terminated, next_truncated, info = step_out
                next_done_np = np.logical_or(next_terminated, next_truncated)
            elif len(step_out) == 4:
                next_obs, reward, next_done_np, info = step_out
            else:
                raise RuntimeError(f"Expected env.step() to return 4 or 5 values, got {len(step_out)}")

            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            if args.action_history_len > 0:
                # The next observation should receive the previous continuous actions
                # from the stacked-frame window. If an environment reset, zero its
                # history so the new episode does not inherit terminal actions.
                next_action_history = torch.roll(next_action_history, shifts=-1, dims=1)
                next_action_history[:, -1] = action.detach()
                done_mask = next_done.bool()
                if done_mask.any():
                    next_action_history[done_mask] = 0.0

            # CarRacing has no Atari lives. Log episode stats whenever an env is done.
            for idx, d in enumerate(next_done_np):
                if d:
                    episodic_return = float(info["r"][idx])
                    episodic_length = int(info["l"][idx])
                    print(f"global_step={global_step}, episodic_return={episodic_return}, std={torch.exp(agent.actor_log_std.data)}")
                    avg_returns.append(episodic_return)
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/avg_episodic_return", float(np.average(avg_returns)), global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        if args.debug_optimizer_logging and args.debug_light_interval > 0 and iteration % args.debug_light_interval == 0:
            log_action_behavior_debug(writer, actions, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_action_history).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_action_histories = action_histories.reshape(
            (-1, args.action_history_len) + envs.single_action_space.shape
        )
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if args.debug_optimizer_logging and args.debug_light_interval > 0 and iteration % args.debug_light_interval == 0:
            log_policy_sharpness_debug(
                writer,
                agent,
                b_obs,
                b_action_histories,
                global_step,
                sample_size=args.debug_policy_sample_size,
            )

            with torch.no_grad():
                writer.add_scalar("debug_value/values_mean", b_values.mean().item(), global_step)
                writer.add_scalar("debug_value/values_std", b_values.std().item(), global_step)
                writer.add_scalar("debug_value/returns_mean", b_returns.mean().item(), global_step)
                writer.add_scalar("debug_value/returns_std", b_returns.std().item(), global_step)
                writer.add_scalar("debug_value/advantages_raw_mean", b_advantages.mean().item(), global_step)
                writer.add_scalar("debug_value/advantages_raw_std", b_advantages.std().item(), global_step)
                writer.add_scalar("debug_value/advantages_abs_mean", b_advantages.abs().mean().item(), global_step)
                if b_values.numel() > 1 and b_values.std() > 1e-8 and b_returns.std() > 1e-8:
                    corr = torch.corrcoef(torch.stack([b_values, b_returns]))[0, 1]
                    writer.add_scalar("debug_value/value_return_corr", corr.item(), global_step)

            log_layernorm_debug(writer, agent, global_step)

        do_delta_debug = (
            args.debug_optimizer_logging
            and args.debug_delta_interval > 0
            and iteration % args.debug_delta_interval == 0
        )
        tracked_before = snapshot_tracked_params(agent, tracked_param_names) if do_delta_debug else {}

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        ratio_abs_devs = []
        ratio_stds = []
        logprob_delta_abses = []
        logprob_delta_means = []
        policy_term_abses = []
        entropy_term_abses = []
        policy_to_entropy_ratios = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_action_histories[mb_inds],
                    b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    if args.debug_optimizer_logging and args.debug_light_interval > 0 and iteration % args.debug_light_interval == 0:
                        ratio_abs_devs.append((ratio - 1.0).abs().mean().item())
                        ratio_stds.append(ratio.std().item())
                        logprob_delta_abses.append(logratio.abs().mean().item())
                        logprob_delta_means.append(logratio.mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                if args.debug_optimizer_logging and args.debug_light_interval > 0 and iteration % args.debug_light_interval == 0:
                    with torch.no_grad():
                        policy_abs = pg_loss.detach().abs().item()
                        entropy_abs = (ent_coef * entropy_loss.detach()).abs().item()
                        policy_term_abses.append(policy_abs)
                        entropy_term_abses.append(entropy_abs)
                        policy_to_entropy_ratios.append(policy_abs / (entropy_abs + 1e-12))

                optimizer.zero_grad()
                loss.backward()

                if not torch.isfinite(loss):
                    print("Non-finite loss detected:", loss.item())
                    print("pg_loss:", pg_loss.item())
                    print("v_loss:", v_loss.item())
                    print("entropy_loss:", entropy_loss.item())
                    raise RuntimeError("Stopping because loss became non-finite.")

                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

                for name, param in agent.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise RuntimeError(f"Non-finite gradient in {name}")

                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        if do_delta_debug:
            log_param_delta_debug(writer, agent, tracked_before, global_step)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if args.debug_optimizer_logging and args.debug_light_interval > 0 and iteration % args.debug_light_interval == 0:
            if ratio_abs_devs:
                writer.add_scalar("debug_policy/ratio_abs_dev", float(np.mean(ratio_abs_devs)), global_step)
                writer.add_scalar("debug_policy/ratio_std", float(np.mean(ratio_stds)), global_step)
                writer.add_scalar("debug_policy/logprob_delta_abs", float(np.mean(logprob_delta_abses)), global_step)
                writer.add_scalar("debug_policy/logprob_delta_mean", float(np.mean(logprob_delta_means)), global_step)
            if policy_to_entropy_ratios:
                writer.add_scalar("debug_loss/policy_term_abs", float(np.mean(policy_term_abses)), global_step)
                writer.add_scalar("debug_loss/entropy_term_abs", float(np.mean(entropy_term_abses)), global_step)
                writer.add_scalar("debug_loss/policy_to_entropy_ratio", float(np.mean(policy_to_entropy_ratios)), global_step)
                writer.add_scalar("debug_loss/current_ent_coef", float(ent_coef), global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        should_eval = (
            args.eval_interval_timesteps > 0
            and (global_step - last_eval_step) >= args.eval_interval_timesteps
        )
        if should_eval:
            last_eval_step = global_step
            eval_seed = args.seed + args.eval_seed_offset + int(global_step)
            eval_mean_return, eval_std_return, eval_returns = evaluate_deterministic_policy(
                agent,
                args,
                device,
                num_episodes=args.eval_episodes,
                num_envs=args.eval_num_envs,
                seed=eval_seed,
            )
            print(
                f"EVAL global_step={global_step}, "
                f"mean_return={eval_mean_return:.3f}, "
                f"std_return={eval_std_return:.3f}, "
                f"returns={np.array(eval_returns, dtype=np.float32)}"
            )
            writer.add_scalar("eval/mean_return", eval_mean_return, global_step)
            writer.add_scalar("eval/std_return", eval_std_return, global_step)
            writer.add_scalar("eval/min_return", float(np.min(eval_returns)), global_step)
            writer.add_scalar("eval/max_return", float(np.max(eval_returns)), global_step)

            if eval_mean_return > best_eval_return:
                best_eval_return = eval_mean_return
                best_eval_returns = eval_returns
                save_checkpoint(
                    best_save_path,
                    agent,
                    optimizer,
                    args,
                    global_step,
                    run_name,
                    best_eval_return=best_eval_return,
                    eval_returns=best_eval_returns,
                    is_best=True,
                )
                abs_best_path = os.path.abspath(best_save_path)
                print(
                    f"New best eval checkpoint: mean_return={best_eval_return:.3f}, "
                    f"path={abs_best_path}"
                )
                writer.add_text("checkpoint/best_agent_path", abs_best_path, global_step)
                writer.add_scalar("eval/best_mean_return", best_eval_return, global_step)

    # If we have a best eval checkpoint, load it before the final 300-track eval.
    if best_eval_returns is not None and os.path.exists(best_save_path):
        checkpoint = torch.load(best_save_path, map_location=device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        print(f"Loaded best checkpoint for final eval: {os.path.abspath(best_save_path)}")
    else:
        print("No best eval checkpoint existed; using final online agent for final eval.")

    final_eval_seed = args.seed + args.eval_seed_offset + 123_456
    final_eval_mean, final_eval_std, final_eval_returns = evaluate_deterministic_policy(
        agent,
        args,
        device,
        num_episodes=args.final_eval_episodes,
        num_envs=args.final_eval_num_envs,
        seed=final_eval_seed,
    )
    final_eval_stats = {
        "num_episodes": args.final_eval_episodes,
        "num_envs": args.final_eval_num_envs,
        "eval_seed": final_eval_seed,
        "checkpoint_path": os.path.abspath(best_save_path) if best_eval_returns is not None and os.path.exists(best_save_path) else None,
        "mean_return": final_eval_mean,
        "std_return": final_eval_std,
        "min_return": float(np.min(final_eval_returns)) if final_eval_returns else float("nan"),
        "max_return": float(np.max(final_eval_returns)) if final_eval_returns else float("nan"),
        "returns": final_eval_returns,
    }
    print(
        f"FINAL_EVAL episodes={args.final_eval_episodes}, "
        f"num_envs={args.final_eval_num_envs}, "
        f"mean_return={final_eval_mean:.3f}, std_return={final_eval_std:.3f}, "
        f"min_return={final_eval_stats['min_return']:.3f}, "
        f"max_return={final_eval_stats['max_return']:.3f}"
    )
    if args.capture_video:
        final_video_seed = args.seed + args.eval_seed_offset + args.final_video_seed_offset
        try:
            video_path, video_return, video_length = record_deterministic_video(
                agent,
                args,
                device,
                video_path=args.final_video_path,
                seed=final_video_seed,
                fps=args.final_video_fps,
                max_steps=args.final_video_max_steps,
            )
            final_eval_stats["video_path"] = video_path
            final_eval_stats["video_return"] = video_return
            final_eval_stats["video_length"] = video_length
            final_eval_stats["video_seed"] = final_video_seed
            print(
                f"FINAL_VIDEO path={video_path}, "
                f"return={video_return:.3f}, length={video_length}, seed={final_video_seed}"
            )
            writer.add_text("final_eval/video_path", video_path, global_step)
            writer.add_scalar("final_eval/video_return", video_return, global_step)
            writer.add_scalar("final_eval/video_length", video_length, global_step)
        except Exception as e:
            print(f"WARNING: final video recording failed: {repr(e)}")

    final_json_path, final_csv_path = write_final_eval_results(
        args.final_eval_results_path,
        final_eval_stats,
        args,
        run_name,
        global_step,
    )
    print(f"Wrote final eval JSON summary to: {final_json_path}")
    print(f"Wrote final eval per-track returns CSV to: {final_csv_path}")

    writer.add_text("final_eval/results_json_path", final_json_path, global_step)
    writer.add_text("final_eval/returns_csv_path", final_csv_path, global_step)
    writer.add_scalar("final_eval/mean_return", final_eval_mean, global_step)
    writer.add_scalar("final_eval/std_return", final_eval_std, global_step)
    writer.add_scalar("final_eval/min_return", final_eval_stats["min_return"], global_step)
    writer.add_scalar("final_eval/max_return", final_eval_stats["max_return"], global_step)

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    save_checkpoint(
        args.save_path,
        agent,
        optimizer,
        args,
        global_step,
        run_name,
        best_eval_return=best_eval_return if best_eval_returns is not None else None,
        eval_returns=best_eval_returns,
        is_best=best_eval_returns is not None,
        final_eval_stats=final_eval_stats,
    )
    abs_save_path = os.path.abspath(args.save_path)
    print(f"Saved final checkpoint with final eval stats to: {abs_save_path}")
    writer.add_text("checkpoint/final_agent_path", abs_save_path, global_step)

    envs.close()
    writer.close()

    if args.track:
        import wandb
        wandb.finish()
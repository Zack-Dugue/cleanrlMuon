# docs: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
import random
import time
import math
from dataclasses import dataclass, fields
from collections import deque
from typing import Dict, Any

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from mpmath import beta
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# your modules
from optimizers import *
from models import Agent, SimpleAgent, ConvSimpleAgent  # <- uses your Agent class

class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.t = 0
        self.ema = 0

    def update(self,value):
        self.ema = self.ema*self.beta + value*(1-self.beta)
        self.t += 1

    def get(self):
        return self.ema / (1-self.beta**self.t)




# ------------------ small wrapper to mimic CleanRL stats ------------------
class RecordEpisodeStatistics(gym.Wrapper):
    """
    EnvPool doesn't have Gymnasium's vector final_info like CleanRL;
    this wrapper accumulates per-env episodic returns/lengths and exposes
    them via info dict keys 'r' and 'l'. We also pass through 'lives' if present.
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        self.returned_episode_returns = None
        self.returned_episode_lengths = None

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # EnvPool provides vectorized info arrays; rewards also live there
        rew = info["reward"] if "reward" in info else reward
        self.episode_returns += rew
        self.episode_lengths += 1

        # expose "current" values for logging each step
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths

        # reset counters where a real episode ended
        if "terminated" in info:
            term = info["terminated"].astype(np.int32)
            self.episode_returns *= (1 - term)
            self.episode_lengths *= (1 - term)

        info["r"] = self.returned_episode_returns
        info["l"] = self.returned_episode_lengths
        # pass through lives if present (ALE often provides it)
        if "lives" not in info:
            info["lives"] = np.zeros(self.num_envs, dtype=np.int32)
        return obs, reward, done, info





# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer





def _safe_tb_name(name: str) -> str:
    """Make a parameter/module name safe-ish for TensorBoard tag paths."""
    return name.replace(".", "_").replace("/", "_")


TRACKED_PARAM_SUFFIXES = (
    "conv1.weight",
    "conv2.weight",
    "conv3.weight",
    "trunk_fc.weight",
    "actor.weight",
    "critic.weight",
    "actor_out.weight",
    "critic_out.weight",
    "actor_mean.weight",
    "critic_out.weight",
)


def _should_track_param(name: str) -> bool:
    return any(name == suffix or name.endswith("." + suffix) for suffix in TRACKED_PARAM_SUFFIXES)


def _snapshot_tracked_params(agent: nn.Module) -> dict[str, torch.Tensor]:
    """
    Snapshot only a few important parameters. This is intentionally sparse because
    cloning every parameter every PPO iteration would add unnecessary overhead.
    """
    return {
        name: p.detach().clone()
        for name, p in agent.named_parameters()
        if _should_track_param(name)
    }


@torch.no_grad()
def _log_update_deltas(
    writer: SummaryWriter,
    agent: nn.Module,
    tracked_before: dict[str, torch.Tensor],
    global_step: int,
):
    """
    Log actual parameter movement after the PPO update:
      delta_norm / param_norm

    This is the most direct way to compare Adam vs Muon, because nominal LR is
    not the same thing as actual update size.
    """
    if not tracked_before:
        return

    for name, p in agent.named_parameters():
        if name not in tracked_before:
            continue

        before = tracked_before[name]
        delta = p.detach() - before

        param_norm = p.detach().norm().item()
        delta_norm = delta.norm().item()
        grad_norm = 0.0 if p.grad is None else p.grad.detach().norm().item()
        safe_name = _safe_tb_name(name)

        writer.add_scalar(f"debug_update/{safe_name}/param_norm", param_norm, global_step)
        writer.add_scalar(f"debug_update/{safe_name}/delta_norm", delta_norm, global_step)
        writer.add_scalar(f"debug_update/{safe_name}/grad_norm", grad_norm, global_step)
        writer.add_scalar(
            f"debug_update/{safe_name}/delta_over_param",
            delta_norm / (param_norm + 1e-12),
            global_step,
        )
        writer.add_scalar(
            f"debug_update/{safe_name}/delta_over_grad",
            delta_norm / (grad_norm + 1e-12),
            global_step,
        )


@torch.no_grad()
def _try_get_logits(agent: nn.Module, x: torch.Tensor):
    """
    Best-effort helper for discrete-policy debug logging.

    Different local Agent/ConvSimpleAgent versions name the actor path
    differently, so this tries common CleanRL-style patterns. If none work,
    it returns None and the main training loop simply skips logit/prob stats.
    """
    # Explicit helpers, if your model defines them.
    for method_name in ("get_logits", "get_action_logits"):
        method = getattr(agent, method_name, None)
        if callable(method):
            try:
                return method(x)
            except Exception:
                pass

    # PQN/ConvSimpleAgent-style helpers.
    features = None
    for feat_name in ("_features", "features"):
        feat_method = getattr(agent, feat_name, None)
        if callable(feat_method):
            try:
                features = feat_method(x)
                break
            except Exception:
                pass

    if features is not None:
        for actor_name in ("actor_out", "actor", "policy", "actor_mean"):
            actor = getattr(agent, actor_name, None)
            if callable(actor):
                try:
                    out = actor(features)
                    if out.ndim >= 2:
                        return out
                except Exception:
                    pass

    # CleanRL Atari Agent usually has self.network and self.actor.
    network = getattr(agent, "network", None)
    actor = getattr(agent, "actor", None)
    if callable(network) and callable(actor):
        for inp in (x.float() / 255.0, x):
            try:
                hidden = network(inp)
                logits = actor(hidden)
                if logits.ndim >= 2:
                    return logits
            except Exception:
                pass

    return None


@torch.no_grad()
def _log_discrete_policy_shape_stats(
    writer: SummaryWriter,
    agent: nn.Module,
    sample_obs: torch.Tensor,
    global_step: int,
):
    """
    Log policy decisiveness stats for discrete-action agents.

    If logits cannot be recovered from the model, this silently skips.
    """
    logits = _try_get_logits(agent, sample_obs)
    if logits is None:
        return

    probs = torch.softmax(logits, dim=-1)
    log_probs_all = torch.log_softmax(logits, dim=-1)

    max_action_prob = probs.max(dim=-1).values.mean()
    logit_std = logits.std()
    logit_norm = logits.norm(dim=-1).mean()
    prob_std = probs.std(dim=-1).mean()

    n_actions = logits.shape[-1]
    uniform_logprob = -math.log(n_actions)
    kl_to_uniform = (probs * (log_probs_all - uniform_logprob)).sum(dim=-1).mean()

    writer.add_scalar("debug_policy/max_action_prob", max_action_prob.item(), global_step)
    writer.add_scalar("debug_policy/logit_std", logit_std.item(), global_step)
    writer.add_scalar("debug_policy/logit_norm", logit_norm.item(), global_step)
    writer.add_scalar("debug_policy/prob_std", prob_std.item(), global_step)
    writer.add_scalar("debug_policy/kl_to_uniform", kl_to_uniform.item(), global_step)


@torch.no_grad()
def _log_action_behavior(
    writer: SummaryWriter,
    actions: torch.Tensor,
    n_actions: int,
    global_step: int,
):
    """
    Log rollout action behavior. For Atari, repeated actions / action fractions
    are often more interpretable than loss alone.
    """
    if actions.numel() == 0:
        return

    if actions.shape[0] > 1:
        repeat_rate = (actions[1:] == actions[:-1]).float().mean().item()
        writer.add_scalar("debug_behavior/action_repeat_rate", repeat_rate, global_step)
        writer.add_scalar("debug_behavior/action_switch_rate", 1.0 - repeat_rate, global_step)

    flat_actions = actions.reshape(-1).detach().long().cpu()
    for a in range(int(n_actions)):
        frac = (flat_actions == a).float().mean().item()
        writer.add_scalar(f"debug_behavior/action_frac/{a}", frac, global_step)


@torch.no_grad()
def _log_value_advantage_stats(
    writer: SummaryWriter,
    b_values: torch.Tensor,
    b_returns: torch.Tensor,
    b_advantages: torch.Tensor,
    global_step: int,
):
    writer.add_scalar("debug_value/values_mean", b_values.mean().item(), global_step)
    writer.add_scalar("debug_value/values_std", b_values.std().item(), global_step)
    writer.add_scalar("debug_value/returns_mean", b_returns.mean().item(), global_step)
    writer.add_scalar("debug_value/returns_std", b_returns.std().item(), global_step)
    writer.add_scalar("debug_value/advantages_raw_mean", b_advantages.mean().item(), global_step)
    writer.add_scalar("debug_value/advantages_raw_std", b_advantages.std().item(), global_step)
    writer.add_scalar("debug_value/advantages_abs_mean", b_advantages.abs().mean().item(), global_step)

    v = b_values.flatten()
    r = b_returns.flatten()
    if v.numel() > 1 and v.std() > 1e-8 and r.std() > 1e-8:
        corr = torch.corrcoef(torch.stack([v, r]))[0, 1]
        writer.add_scalar("debug_value/value_return_corr", corr.item(), global_step)



# ----------------------------- Args -----------------------------
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = False
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "corr_ppo_project"
    wandb_entity: str = None
    wandb_tag: str = None
    capture_video: bool = False  # EnvPool path doesn’t record videos by default

    #multi_gpu stuff:
    device: str = None

    # Algorithm
    env_id: str = "Breakout-v5"  # EnvPool ALE v5 id
    total_timesteps: int = 10_000_000
    learning_rate: float = 1e-3
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.5
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # Optimizer parity with your PPO Atari script
    optimizer: str = "Adam"  # ["SGD", "Adam", "Muon", "AdaMuon"]
    momentum: float = 0.9
    aux_learning_rate: float | None = None
    """optional aux Adam LR for Muon-style optimizers; defaults to learning_rate / 300"""

    # Debug logging. Cheap policy/action stats can be logged every PPO iteration.
    # Expensive parameter-delta stats should be sparse.
    debug_policy_every: int = 1
    """log cheap policy/ratio/action/value diagnostics every N PPO iterations; <=0 disables"""
    debug_update_every: int = 50
    """log expensive parameter delta diagnostics every N PPO iterations; <=0 disables"""
    debug_log_action_fractions: bool = True
    """log per-action rollout frequencies"""
    debug_log_policy_shape: bool = True
    """try to log logits/probability shape stats such as max_action_prob and logit_std"""

    EV_beta: float = .95
    """Get the beta of the explained variance"""

    use_correlation_weighting: bool = True
    """If enabled, scale PPO clipping by the EMA value/return correlation score; if disabled, use scale=1.0."""

    correlation_scale_floor: float = 0.01
    """Minimum clip/value-clip scale when correlation weighting is enabled."""

    # runtime-filled
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def _str2bool(v):
    """
    argparse-compatible bool parser that accepts:
      --flag
      --flag=true
      --flag=True
      --flag false
      --no-flag
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    v = str(v).strip().lower()
    if v in ("yes", "true", "t", "1", "y", "on"):
        return True
    if v in ("no", "false", "f", "0", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {v!r}")


def _arg_type_for_field(name, default):
    """
    Infer argparse type from the dataclass default. For None defaults,
    use a small explicit map matching this script's optional args.
    """
    if isinstance(default, bool):
        return _str2bool
    if isinstance(default, int) and not isinstance(default, bool):
        return int
    if isinstance(default, float):
        return float
    if default is None:
        none_type_map = {
            "wandb_entity": str,
            "wandb_tag": str,
            "device": str,
            "target_kl": float,
            "aux_learning_rate": float,
        }
        return none_type_map.get(name, str)
    return type(default)


def parse_args() -> Args:
    """
    Replacement for tyro.cli(Args).

    This deliberately accepts explicit bool values because the HPO launcher emits
    args like --use-correlation-weighting=True/False. It also accepts normal
    presence/absence flags and --no-* negations.
    """
    parser = argparse.ArgumentParser()
    defaults = Args()

    for f in fields(Args):
        name = f.name
        default = getattr(defaults, name)
        cli_name = "--" + name.replace("_", "-")

        if isinstance(default, bool):
            parser.add_argument(
                cli_name,
                nargs="?",
                const=True,
                default=default,
                type=_str2bool,
            )
            parser.add_argument(
                "--no-" + name.replace("_", "-"),
                dest=name,
                action="store_false",
            )
        else:
            parser.add_argument(
                cli_name,
                default=default,
                type=_arg_type_for_field(name, default),
            )

    ns = parser.parse_args()
    return Args(**vars(ns))

#Need to fix the whole 'unpicklable nested function' thing BUT
# We should ditch the gpu environment thing, and
# Just directly set the GPU and the number of threads in multigpu_tuner_spv.

# ----------------------------- main -----------------------------
if __name__ == "__main__":
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            tags=[args.wandb_tag] if args.wandb_tag else None,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    # Seeding / determinism
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ---------------- EnvPool setup ----------------
    # NOTE: EnvPool handles Atari preprocessing internally.
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    # Provide expected shape fields like CleanRL’s vector envs
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)

    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action spaces supported"

    # model
    # agent = BetterSimpleAgent(envs,use_muon_input=True).to(device)
    # agent = Agent(envs).to(device)
    agent = ConvSimpleAgent(envs,use_muon_input=True).to(device)
    MC_Method = False

    # -------- Optimizer selection (mirrors your PPO Atari code) --------
    device_type = device.type
    aux_lr = args.aux_learning_rate if args.aux_learning_rate is not None else args.learning_rate / 300
    if args.optimizer == "SGD":
        optimizer = optim.SGD(agent.parameters(), momentum=args.momentum, lr=args.learning_rate)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(agent.parameters(), betas=(args.momentum, 0.999), lr=args.learning_rate, eps=1e-5)
    elif args.optimizer == "Muon":
        muon_params, aux_params = agent.get_split_params()
        ns_steps = 2 if device_type == "cpu" else 5
        param_groups = [
            dict(params=muon_params, lr=args.learning_rate, momentum=args.momentum,
                 weight_decay=1e-4, use_muon=True, ns_steps=ns_steps),
            dict(params=aux_params, lr=aux_lr, momentum=args.momentum,
                 weight_decay=1e-4, use_muon=False),
        ]
        optimizer = MuonWithAuxAdam(param_groups)

    elif args.optimizer == "RLMuon":
        muon_params, aux_params = agent.get_split_params()
        ns_steps = 2 if device_type == "cpu" else 5
        param_groups = [
            dict(params=muon_params, lr=args.learning_rate, momentum=args.momentum, var_momentum=args.momentum**.25,
                 weight_decay=1e-4, use_muon=True, nesterov=True, ns_steps=ns_steps),
            dict(params=aux_params, lr=aux_lr,
                 weight_decay=1e-4, use_muon=False),
        ]
        optimizer = RLMuonWithAuxAdam(param_groups)
    elif args.optimizer == "NorMuon":
        muon_params, aux_params = agent.get_split_params()
        param_groups = [
            dict(params=muon_params, lr=args.learning_rate, weight_decay=1e-4, use_muon=True),
            dict(params=aux_params, lr=aux_lr, weight_decay=1e-4, use_muon=False),
        ]
        optimizer = SingleDeviceNorMuonWithAuxAdam(param_groups)
    elif args.optimizer == "AdaMuon":
        muon_params, aux_params = agent.get_split_params()
        param_groups = [
            dict(params=muon_params, lr=args.learning_rate, weight_decay=1e-4, use_muon=True),
            dict(params=aux_params, lr=aux_lr, weight_decay=1e-4, use_muon=False),
        ]
        optimizer = AdaMuonWithAuxAdam(param_groups)
    elif args.optimizer == "BGD":
        params = BGD.create_unique_param_groups(agent)
        optimizer = BGD(params, std_init=.01, mean_eta=args.learning_rate, std_eta=10,
                        betas=(args.momentum, .999, .99), mc_iters=1)
        MC_Method = True


    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # -------- Storage --------
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)  # float accumulator
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # -------- Rollout loop --------
    global_step = 0
    start_time = time.time()

    next_obs = torch.tensor(envs.reset(), device=device)
    # Keep as float (0.0 alive, 1.0 done) to avoid bool-minus ops later
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.float32)

    avg_returns = deque(maxlen=20)

    # Print/debug what params we will track for optimizer movement.
    tracked_param_names = [name for name, _ in agent.named_parameters() if _should_track_param(name)]
    print(f"[debug] tracking parameter deltas for: {tracked_param_names}")
    writer.add_text("debug/tracked_params", "\n".join(tracked_param_names) if tracked_param_names else "(none)", 0)

    var_true_ema = EMA(args.EV_beta)
    var_estimate_ema = EMA(args.EV_beta)
    covar_ema = EMA(args.EV_beta)
    scale = 1

    for iteration in range(1, args.num_iterations + 1):
        # LR anneal across all groups
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for g in optimizer.param_groups:
                g["lr"] = lrnow
        if MC_Method:
            optimizer.randomize_weights(force_std=0)
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # EnvPool step
            nobs, reward, ndone, info = envs.step(action.detach().cpu().numpy())
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.tensor(nobs, device=device)

            # ndone is a boolean array from EnvPool; store as float 0/1
            next_done = torch.tensor(ndone, device=device, dtype=torch.float32)

            # When a life-based episode terminates, log return/length
            # Here, a "true episode" ended if lives==0 and done==True (ALE terminology)
            # We just follow the prior wrapper’s 'r' and 'l' arrays.
            term_idx = np.where(ndone)[0]
            for idx in term_idx:
                # Only log at true episode boundaries — EnvPool gives 'lives' in info if enabled
                if "lives" in info and info["lives"][idx] == 0:
                    ret = float(info["r"][idx])
                    length = int(info["l"][idx])
                    print(f"global_step={global_step}, episodic_return={ret}, scale={scale}")
                    avg_returns.append(ret)
                    writer.add_scalar("charts/avg_episodic_return", float(np.average(avg_returns)), global_step)
                    writer.add_scalar("charts/episodic_return", ret, global_step)
                    writer.add_scalar("charts/episodic_length", length, global_step)

        # ------- cheap rollout behavior debug logging -------
        do_debug_policy = args.debug_policy_every > 0 and (iteration % args.debug_policy_every == 0)
        do_debug_update = args.debug_update_every > 0 and (iteration % args.debug_update_every == 0)

        if do_debug_policy:
            if args.debug_log_action_fractions:
                _log_action_behavior(writer, actions, envs.single_action_space.n, global_step)

        # Snapshot selected parameters once before the whole PPO update, not every minibatch.
        tracked_before = _snapshot_tracked_params(agent) if do_debug_update else {}

        # ------- GAE -------
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done  # float tensor
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)


        with torch.no_grad():
            y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
            var_true_ema.update(y_true.var())
            var_estimate_ema.update(y_pred.var())
            covar_ema.update((y_true*y_pred).mean() - (y_true.mean()*y_pred.mean()))

            corr_denom = math.sqrt(max(var_true_ema.get() * var_estimate_ema.get(), 1e-12))
            value_return_corr_ema = covar_ema.get() / corr_denom
            correlation_scale = max(args.correlation_scale_floor, value_return_corr_ema**2)
            scale = correlation_scale if args.use_correlation_weighting else 1.0

        writer.add_scalar("correlation_weighting/value_return_corr_ema", value_return_corr_ema, global_step)
        writer.add_scalar("correlation_weighting/correlation_scale", correlation_scale, global_step)
        writer.add_scalar("correlation_weighting/effective_clip_scale", scale, global_step)
        writer.add_scalar("correlation_weighting/enabled", float(args.use_correlation_weighting), global_step)

        if do_debug_policy:
            _log_value_advantage_stats(writer, b_values, b_returns, b_advantages, global_step)

        # ------- PPO updates -------
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        # Aggregate minibatch-level debug stats and write means once per PPO iteration.
        dbg_ratio_means = []
        dbg_ratio_stds = []
        dbg_ratio_abs_devs = []
        dbg_logprob_delta_abs = []
        dbg_logprob_delta_mean = []
        dbg_policy_to_entropy = []
        dbg_policy_term_abs = []
        dbg_entropy_term_abs = []
        dbg_value_term_abs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                if MC_Method:
                    optimizer.randomize_weights()
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef*scale).float().mean().item())

                    if do_debug_policy:
                        dbg_ratio_means.append(ratio.mean().item())
                        dbg_ratio_stds.append(ratio.std().item())
                        dbg_ratio_abs_devs.append((ratio - 1.0).abs().mean().item())
                        dbg_logprob_delta_abs.append(logratio.abs().mean().item())
                        dbg_logprob_delta_mean.append(logratio.mean().item())

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef*scale, 1 + args.clip_coef*scale)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds],
                                                                -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()


                loss = (pg_loss - args.ent_coef * entropy_loss) + v_loss * args.vf_coef

                if do_debug_policy:
                    with torch.no_grad():
                        entropy_term = args.ent_coef * entropy_loss
                        value_term = args.vf_coef * v_loss
                        dbg_policy_term_abs.append(pg_loss.abs().item())
                        dbg_entropy_term_abs.append(entropy_term.abs().item())
                        dbg_value_term_abs.append(value_term.abs().item())
                        dbg_policy_to_entropy.append(
                            pg_loss.abs().item() / (entropy_term.abs().item() + 1e-12)
                        )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                if MC_Method:
                    optimizer.aggregate_grads(1)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        if do_debug_update:
            _log_update_deltas(writer, agent, tracked_before, global_step)

        # Log policy-shape stats once per PPO iteration on one minibatch-sized sample.
        if do_debug_policy and args.debug_log_policy_shape:
            sample_n = min(args.minibatch_size, b_obs.shape[0])
            _log_discrete_policy_shape_stats(writer, agent, b_obs[:sample_n], global_step)

        # logging
        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)) if clipfracs else 0.0, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if do_debug_policy:
            if dbg_ratio_means:
                writer.add_scalar("debug_policy/ratio_mean", float(np.mean(dbg_ratio_means)), global_step)
                writer.add_scalar("debug_policy/ratio_std", float(np.mean(dbg_ratio_stds)), global_step)
                writer.add_scalar("debug_policy/ratio_abs_dev", float(np.mean(dbg_ratio_abs_devs)), global_step)
                writer.add_scalar("debug_policy/logprob_delta_abs", float(np.mean(dbg_logprob_delta_abs)), global_step)
                writer.add_scalar("debug_policy/logprob_delta_mean", float(np.mean(dbg_logprob_delta_mean)), global_step)
            if dbg_policy_to_entropy:
                writer.add_scalar("debug_loss/policy_term_abs", float(np.mean(dbg_policy_term_abs)), global_step)
                writer.add_scalar("debug_loss/entropy_term_abs", float(np.mean(dbg_entropy_term_abs)), global_step)
                writer.add_scalar("debug_loss/value_term_abs", float(np.mean(dbg_value_term_abs)), global_step)
                writer.add_scalar("debug_loss/policy_to_entropy_ratio", float(np.mean(dbg_policy_to_entropy)), global_step)

        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

    # cleanup
    envs.close()
    writer.close()
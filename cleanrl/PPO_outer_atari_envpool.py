# docs/source inspiration:
#   CleanRL PPO Atari EnvPool style:
#   https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
#
# ============================================================
# PPO Atari EnvPool with Inner Muon / Outer AdamW
# ============================================================
#
# Idea:
#   1. Collect rollout with the outer/deployed policy agent θ_old.
#   2. Clone θ_old into an inner agent φ.
#   3. Run aggressive inner PPO optimization on φ using Muon.
#      - Inner LR decays within the update phase itself.
#      - Inner optimizer uses decoupled AdamW-style decay toward θ_old:
#
#            p <- p - lr * old_weight_decay * (p - p_old)
#
#   4. Compute parameter delta:
#
#            delta = φ_K - θ_old
#
#   5. Treat θ_old - φ_K as pseudo-gradient for outer AdamW:
#
#            θ.grad = θ_old - φ_K
#            outer_adamw.step()
#
# Notes:
#   - No KL penalty.
#   - No line search.
#   - No extra regularizer beyond PPO's normal clipped objective/value/entropy
#     and the requested inner old-weight proximal decay.
#   - outer_weight_decay defaults to 0.0 to avoid adding another regularizer.
# ============================================================

import argparse
import copy
import math
import os
import random
import time
from collections import deque

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# Args
# ============================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment args
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--track", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL")
    parser.add_argument("--wandb-entity", type=str, default=None)

    # Env / PPO args
    parser.add_argument("--env-id", type=str, default="Pong-v5")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)

    parser.add_argument("--anneal-outer-lr", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--gae", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)

    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--norm-adv", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--clip-vloss", type=str2bool, default=True, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Optional early stop only. Default off.
    parser.add_argument("--target-kl", type=float, default=None)

    # Outer optimizer
    parser.add_argument("--outer-learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--outer-betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--outer-eps", type=float, default=1e-5)
    parser.add_argument("--outer-weight-decay", type=float, default=0.0)

    # Inner Muon optimizer
    parser.add_argument("--inner-muon-lr", type=float, default=0.05)
    parser.add_argument("--inner-aux-lr", type=float, default=1e-3)
    parser.add_argument("--inner-momentum", type=float, default=0.95)
    parser.add_argument("--inner-aux-beta2", type=float, default=0.95)
    parser.add_argument("--inner-eps", type=float, default=1e-8)
    parser.add_argument("--inner-ns-steps", type=int, default=5)

    # This is the requested "AdamW-style toward old policy" decay.
    parser.add_argument("--inner-old-weight-decay", type=float, default=0.01)

    # LR decay inside the inner optimization loop.
    # At the final inner minibatch, lr ~= initial_lr * inner_lr_final_frac.
    parser.add_argument("--inner-lr-final-frac", type=float, default=0.10)

    # Muon routing options
    parser.add_argument("--muon-include-heads", type=str2bool, default=False, nargs="?", const=True)

    args = parser.parse_args()

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args


# ============================================================
# Env wrapper from CleanRL-style EnvPool Atari script
# ============================================================

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

        self.has_lives = False
        env.reset()
        info = env.step(np.zeros(self.num_envs, dtype=int))[-1]
        if "lives" in info and info["lives"].sum() > 0:
            self.has_lives = True
            print("env has lives")

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)

        # EnvPool returns reward both as second tuple item and in infos["reward"].
        reward_for_stats = infos["reward"] if "reward" in infos else rewards
        self.episode_returns += reward_for_stats
        self.episode_lengths += 1

        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths

        if self.has_lives and "lives" in infos:
            all_lives_exhausted = infos["lives"] == 0
            self.episode_returns *= 1 - all_lives_exhausted
            self.episode_lengths *= 1 - all_lives_exhausted
        else:
            self.episode_returns *= 1 - dones
            self.episode_lengths *= 1 - dones

        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths

        return observations, rewards, dones, infos


# ============================================================
# Agent
# ============================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def split_muon_and_aux_params(self, include_heads=False, verbose=False):
        """
        Conservative Muon routing:
          Muon:
            network.2.weight  conv2
            network.4.weight  conv3
            network.7.weight  hidden fc

          Aux AdamW:
            network.0.weight  input conv
            actor.weight      output head
            critic.weight     output head
            all biases

        If include_heads=True:
          actor.weight and critic.weight also go to Muon.
        """
        muon_names = {
            "network.2.weight",
            "network.4.weight",
            "network.7.weight",
        }

        if include_heads:
            muon_names |= {
                "actor.weight",
                "critic.weight",
            }

        muon_params = []
        muon_names_actual = []
        aux_params = []
        aux_names_actual = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            if name in muon_names and p.ndim >= 2:
                muon_params.append(p)
                muon_names_actual.append(name)
                if verbose:
                    print("Muon param:", name, tuple(p.shape))
            else:
                aux_params.append(p)
                aux_names_actual.append(name)
                if verbose:
                    print("Aux AdamW param:", name, tuple(p.shape))

        return muon_params, aux_params, muon_names_actual, aux_names_actual


# ============================================================
# Muon utilities
# ============================================================

@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    assert G.ndim == 2, f"zeropower expects 2D tensor, got {tuple(G.shape)}"

    orig_dtype = G.dtype
    X = G.float()
    X = X / (X.norm() + eps)

    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(orig_dtype)


class InnerMuonWithAuxAdamWOldWeightDecay(torch.optim.Optimizer):
    """
    Inner optimizer.

    Muon params:
      - Muon momentum
      - Newton-Schulz orthogonalized update
      - decoupled proximal decay toward old policy:
            p <- p - lr * old_weight_decay * (p - p_old)

    Aux params:
      - AdamW-like adaptive update
      - decoupled proximal decay toward old policy:
            p <- p - lr * old_weight_decay * (p - p_old)

    Important:
      This is not standard weight decay to zero.
      This is weight decay toward the old rollout policy weights.
    """

    def __init__(self, param_groups):
        expanded = []

        for g in param_groups:
            assert "use_muon" in g
            params = list(g["params"])
            old_params = list(g["old_params"])
            assert len(params) == len(old_params)

            if not params:
                continue

            if g["use_muon"]:
                expanded.append(dict(
                    params=params,
                    old_params=old_params,
                    use_muon=True,
                    lr=g.get("lr", 0.05),
                    momentum=g.get("momentum", 0.95),
                    old_weight_decay=g.get("old_weight_decay", 0.01),
                    ns_steps=g.get("ns_steps", 5),
                    eps=g.get("eps", 1e-8),
                ))
            else:
                expanded.append(dict(
                    params=params,
                    old_params=old_params,
                    use_muon=False,
                    lr=g.get("lr", 1e-3),
                    betas=g.get("betas", (0.9, 0.95)),
                    old_weight_decay=g.get("old_weight_decay", 0.01),
                    eps=g.get("eps", 1e-8),
                ))

        super().__init__(expanded, {})

    @torch.no_grad()
    def set_lrs(self, muon_lr, aux_lr):
        for group in self.param_groups:
            if group["use_muon"]:
                group["lr"] = muon_lr
            else:
                group["lr"] = aux_lr

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_aux_adamw_group(group)

        return loss

    @torch.no_grad()
    def _apply_old_weight_decay(self, p, p_old, lr, wd):
        if wd != 0:
            p.add_(p - p_old, alpha=-lr * wd)

    @torch.no_grad()
    def _step_aux_adamw_group(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        old_wd = group["old_weight_decay"]

        for p, p_old in zip(group["params"], group["old_params"]):
            g = p.grad
            if g is None:
                continue

            self._apply_old_weight_decay(p, p_old, lr, old_wd)

            st = self.state[p]
            if len(st) == 0:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)

            st["step"] += 1
            t = st["step"]

            m = st["exp_avg"]
            v = st["exp_avg_sq"]

            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t

            update = (m / bc1) / (v.sqrt() / math.sqrt(bc2) + eps)
            p.add_(update, alpha=-lr)

    @torch.no_grad()
    def _step_muon_group(self, group):
        lr = group["lr"]
        momentum = group["momentum"]
        old_wd = group["old_weight_decay"]
        ns_steps = group["ns_steps"]
        eps = group["eps"]

        for p, p_old in zip(group["params"], group["old_params"]):
            g = p.grad
            if g is None:
                continue

            if p.ndim < 2:
                raise ValueError("Muon group received <2D parameter. Route biases/norms to aux AdamW.")

            self._apply_old_weight_decay(p, p_old, lr, old_wd)

            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(g)

            buf = st["momentum_buffer"]
            buf.mul_(momentum).add_(g)

            # Nesterov-ish Muon direction.
            g_eff = g.add(buf, alpha=momentum)

            original_shape = g_eff.shape
            g_eff_2d = g_eff.contiguous().view(g_eff.shape[0], -1)

            update_2d = zeropower_via_newtonschulz5(g_eff_2d, steps=ns_steps, eps=eps)

            rows, cols = update_2d.shape
            dim_scale = 0.2 * math.sqrt(max(rows, cols))

            update = update_2d.view(original_shape)
            p.add_(update, alpha=-lr * dim_scale)


# ============================================================
# Helper functions
# ============================================================

def inner_lr_multiplier(inner_step, total_inner_steps, final_frac):
    """
    Cosine decay inside one PPO update phase.
    Starts at 1.0 and ends near final_frac.
    """
    if total_inner_steps <= 1:
        return 1.0

    progress = inner_step / float(total_inner_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_frac + (1.0 - final_frac) * cosine


def build_inner_optimizer(inner_agent, old_params_by_name, args, verbose=False):
    muon_params, aux_params, muon_names, aux_names = inner_agent.split_muon_and_aux_params(
        include_heads=args.muon_include_heads,
        verbose=verbose,
    )

    muon_old_params = [old_params_by_name[name] for name in muon_names]
    aux_old_params = [old_params_by_name[name] for name in aux_names]

    optimizer = InnerMuonWithAuxAdamWOldWeightDecay([
        {
            "params": muon_params,
            "old_params": muon_old_params,
            "use_muon": True,
            "lr": args.inner_muon_lr,
            "momentum": args.inner_momentum,
            "old_weight_decay": args.inner_old_weight_decay,
            "ns_steps": args.inner_ns_steps,
            "eps": args.inner_eps,
        },
        {
            "params": aux_params,
            "old_params": aux_old_params,
            "use_muon": False,
            "lr": args.inner_aux_lr,
            "betas": (args.inner_momentum, args.inner_aux_beta2),
            "old_weight_decay": args.inner_old_weight_decay,
            "eps": args.inner_eps,
        },
    ])

    return optimizer


@torch.no_grad()
def set_outer_pseudo_grads_from_inner(agent, inner_agent):
    """
    AdamW minimizes, so use:
        grad = old - inner

    Then AdamW step moves old parameters toward the inner solution.
    """
    total_delta_sq = 0.0
    total_grad_sq = 0.0
    total_params = 0

    for p_outer, p_inner in zip(agent.parameters(), inner_agent.parameters()):
        if not p_outer.requires_grad:
            continue

        delta = p_inner.detach() - p_outer.detach()
        pseudo_grad = -delta

        p_outer.grad = pseudo_grad.clone()

        total_delta_sq += delta.pow(2).sum().item()
        total_grad_sq += pseudo_grad.pow(2).sum().item()
        total_params += delta.numel()

    delta_norm = math.sqrt(total_delta_sq)
    pseudo_grad_norm = math.sqrt(total_grad_sq)
    delta_rms = math.sqrt(total_delta_sq / max(total_params, 1))

    return {
        "inner_outer_delta_norm": delta_norm,
        "outer_pseudo_grad_norm": pseudo_grad_norm,
        "inner_outer_delta_rms": delta_rms,
    }


@torch.no_grad()
def compute_outer_kl_metrics(agent, b_obs, b_actions, b_logprobs, clip_coef, minibatch_size=4096):
    """
    Measurement only. No regularization or line search.
    Computes logratio/KL/clipfrac between rollout policy and current outer policy.
    """
    n = b_obs.shape[0]
    approx_kls = []
    old_approx_kls = []
    clipfracs = []

    for start in range(0, n, minibatch_size):
        end = min(start + minibatch_size, n)
        _, newlogprob, _, _ = agent.get_action_and_value(
            b_obs[start:end],
            b_actions.long()[start:end],
        )
        logratio = newlogprob - b_logprobs[start:end]
        ratio = logratio.exp()

        old_approx_kls.append((-logratio).mean())
        approx_kls.append(((ratio - 1.0) - logratio).mean())
        clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean())

    return {
        "outer_old_approx_kl": torch.stack(old_approx_kls).mean().item(),
        "outer_approx_kl": torch.stack(approx_kls).mean().item(),
        "outer_clipfrac": torch.stack(clipfracs).mean().item(),
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = parse_args()

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
        "|param|value|\n|-|-|\n%s"
        % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("device:", device)

    # Env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)

    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)

    # Outer optimizer. Default weight decay is zero so the only special regularizer
    # is the requested inner old-weight decay.
    outer_optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=args.outer_learning_rate,
        betas=tuple(args.outer_betas),
        eps=args.outer_eps,
        weight_decay=args.outer_weight_decay,
    )

    # Rollout storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    avg_returns = deque(maxlen=20)

    global_step = 0
    start_time = time.time()

    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs, device=device)

    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Outer LR annealing across full training.
        if args.anneal_outer_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            outer_lr_now = frac * args.outer_learning_rate
            for group in outer_optimizer.param_groups:
                group["lr"] = outer_lr_now
        else:
            outer_lr_now = args.outer_learning_rate

        # ====================================================
        # Rollout collection with deployed outer agent θ_old
        # ====================================================
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)

            for idx, d in enumerate(done):
                if d and ("lives" not in info or info["lives"][idx] == 0):
                    episodic_return = info["r"][idx]
                    episodic_length = info["l"][idx]
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    avg_returns.append(episodic_return)
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)

        # ====================================================
        # GAE / returns
        # ====================================================
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)

            if args.gae:
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]

                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )

                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards, device=device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]

                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return

                advantages = returns - values

        # Flatten rollout batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ====================================================
        # Inner loop: clone old policy and aggressively optimize
        # ====================================================
        inner_agent = Agent(envs).to(device)
        inner_agent.load_state_dict(agent.state_dict())

        old_params_by_name = {
            name: p.detach().clone()
            for name, p in inner_agent.named_parameters()
            if p.requires_grad
        }

        inner_optimizer = build_inner_optimizer(
            inner_agent=inner_agent,
            old_params_by_name=old_params_by_name,
            args=args,
            verbose=(update == 1),
        )

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        total_inner_steps = args.update_epochs * args.num_minibatches
        inner_step = 0

        # Logging accumulators for final inner minibatch values
        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        old_approx_kl = torch.tensor(0.0, device=device)
        approx_kl = torch.tensor(0.0, device=device)
        inner_loss = torch.tensor(0.0, device=device)

        stop_inner = False

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                lr_mult = inner_lr_multiplier(
                    inner_step=inner_step,
                    total_inner_steps=total_inner_steps,
                    final_frac=args.inner_lr_final_frac,
                )
                inner_muon_lr_now = args.inner_muon_lr * lr_mult
                inner_aux_lr_now = args.inner_aux_lr * lr_mult
                inner_optimizer.set_lrs(
                    muon_lr=inner_muon_lr_now,
                    aux_lr=inner_aux_lr_now,
                )

                _, newlogprob, entropy, newvalue = inner_agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions.long()[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Clipped policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - args.clip_coef,
                    1.0 + args.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Clipped value loss
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

                # Standard PPO minibatch objective.
                # The requested old-policy weight decay is not added here;
                # it is applied decoupled inside the inner optimizer step.
                inner_loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                inner_optimizer.zero_grad(set_to_none=True)
                inner_loss.backward()
                nn.utils.clip_grad_norm_(inner_agent.parameters(), args.max_grad_norm)
                inner_optimizer.step()

                inner_step += 1

                if args.target_kl is not None and approx_kl > args.target_kl:
                    stop_inner = True
                    break

            if stop_inner:
                break

        # ====================================================
        # Outer loop: use inner delta as pseudo-gradient
        # ====================================================
        outer_optimizer.zero_grad(set_to_none=True)

        delta_stats = set_outer_pseudo_grads_from_inner(agent, inner_agent)

        # This is the actual deployed-policy update.
        outer_optimizer.step()

        # Measurement only. No KL penalty / no line search.
        outer_kl_stats = compute_outer_kl_metrics(
            agent=agent,
            b_obs=b_obs,
            b_actions=b_actions,
            b_logprobs=b_logprobs,
            clip_coef=args.clip_coef,
            minibatch_size=args.minibatch_size,
        )

        # ====================================================
        # Logging
        # ====================================================
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", outer_lr_now, global_step)
        writer.add_scalar("charts/inner_muon_lr_start", args.inner_muon_lr, global_step)
        writer.add_scalar("charts/inner_aux_lr_start", args.inner_aux_lr, global_step)
        writer.add_scalar("charts/inner_muon_lr_final", inner_muon_lr_now, global_step)
        writer.add_scalar("charts/inner_aux_lr_final", inner_aux_lr_now, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/inner_loss", inner_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl_inner_final_mb", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac_inner", np.mean(clipfracs) if len(clipfracs) else 0.0, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        writer.add_scalar("outer/inner_outer_delta_norm", delta_stats["inner_outer_delta_norm"], global_step)
        writer.add_scalar("outer/inner_outer_delta_rms", delta_stats["inner_outer_delta_rms"], global_step)
        writer.add_scalar("outer/pseudo_grad_norm", delta_stats["outer_pseudo_grad_norm"], global_step)
        writer.add_scalar("outer/old_approx_kl_after_outer_step", outer_kl_stats["outer_old_approx_kl"], global_step)
        writer.add_scalar("outer/approx_kl_after_outer_step", outer_kl_stats["outer_approx_kl"], global_step)
        writer.add_scalar("outer/clipfrac_after_outer_step", outer_kl_stats["outer_clipfrac"], global_step)

        print(
            f"update={update}/{num_updates} "
            f"global_step={global_step} "
            f"SPS={int(global_step / (time.time() - start_time))} "
            f"pg_loss={pg_loss.item():.4f} "
            f"v_loss={v_loss.item():.4f} "
            f"entropy={entropy_loss.item():.4f} "
            f"inner_kl={approx_kl.item():.6f} "
            f"outer_kl={outer_kl_stats['outer_approx_kl']:.6f} "
            f"delta_rms={delta_stats['inner_outer_delta_rms']:.6e}"
        )

        del inner_agent
        del inner_optimizer

        if device.type == "cuda":
            torch.cuda.empty_cache()

    envs.close()
    writer.close()
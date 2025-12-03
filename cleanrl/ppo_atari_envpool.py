# docs: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
import random
import time
from dataclasses import dataclass
from collections import deque
from typing import Dict, Any

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# your modules
from optimizers import AdaMuonWithAuxAdam, MuonWithAuxAdam, BGD, SingleDeviceNorMuonWithAuxAdam
from models import Agent  # <- uses your Agent class

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

# ----------------------------- Args -----------------------------
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False  # EnvPool path doesn’t record videos by default

    #multi_gpu stuff:
    device: str = None

    # Algorithm
    env_id: str = "Breakout-v5"  # EnvPool ALE v5 id
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # Optimizer parity with your PPO Atari script
    optimizer: str = "Adam"  # ["SGD", "Adam", "Muon", "AdaMuon"]
    momentum: float = 0.9

    # runtime-filled
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

#Need to fix the whole 'unpicklable nested function' thing BUT
# We should ditch the gpu environment thing, and
# Just directly set the GPU and the number of threads in multigpu_tuner_spv.

# ----------------------------- main -----------------------------
if __name__ == "__main__":
    args = tyro.cli(Args)
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
    agent = Agent(envs).to(device)
    MC_Method = False

    # -------- Optimizer selection (mirrors your PPO Atari code) --------
    device_type = device.type
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
            dict(params=aux_params, lr=args.learning_rate, momentum=args.momentum,
                 weight_decay=1e-4, use_muon=False),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    elif args.optimizer == "NorMuon":
        muon_params, aux_params = agent.get_split_params()
        param_groups = [
            dict(params=muon_params, lr=args.learning_rate, weight_decay=1e-4, use_muon=True),
            dict(params=aux_params, lr=args.learning_rate, weight_decay=1e-4, use_muon=False),
        ]
        optimizer = SingleDeviceNorMuonWithAuxAdam(param_groups)
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
                    print(f"global_step={global_step}, episodic_return={ret}")
                    avg_returns.append(ret)
                    writer.add_scalar("charts/avg_episodic_return", float(np.average(avg_returns)), global_step)
                    writer.add_scalar("charts/episodic_return", ret, global_step)
                    writer.add_scalar("charts/episodic_length", length, global_step)

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

        # ------- PPO updates -------
        b_inds = np.arange(args.batch_size)
        clipfracs = []
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
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef


                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                if MC_Method:
                    optimizer.aggregate_grads(1)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

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

        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

    # cleanup
    envs.close()
    writer.close()

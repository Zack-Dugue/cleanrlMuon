# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/pqn/#pqn_atari_envpoolpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from optimizers import AdaMuonWithAuxAdam, MuonWithAuxAdam, BGD, SingleDeviceNorMuonWithAuxAdam
from models import BetterPQNQNetwork


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
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # optional multi-gpu launcher compatibility
    device: str = None

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the Q-network"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    q_lambda: float = 0.65
    """the lambda for the Q-Learning algorithm"""

    # optimizer parity with your PPO script
    optimizer: str = "Adam"  # ["SGD", "Adam", "Muon", "NorMuon", "AdaMuon", "BGD"]
    momentum: float = 0.9

    # model optimizer-routing kwargs
    use_muon_input: bool = False
    use_muon_output: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        if "lives" not in infos:
            infos["lives"] = np.zeros(self.num_envs, dtype=np.int32)
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if True:
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    print(f"envs.action_space type = {type(envs.action_space)}")
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = BetterPQNQNetwork(
        envs,
        use_muon_input=args.use_muon_input,
        use_muon_output=args.use_muon_output,
    ).to(device)

    MC_Method = False
    device_type = device.type

    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            q_network.parameters(),
            momentum=args.momentum,
            lr=args.learning_rate,
        )
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(
            q_network.parameters(),
            betas=(args.momentum, 0.999),
            lr=args.learning_rate,
            eps=1e-5,
        )
    elif args.optimizer == "Muon":
        muon_params, aux_params = q_network.get_split_params()
        ns_steps = 2 if device_type == "cpu" else 5
        param_groups = [
            dict(
                params=muon_params,
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=1e-4,
                use_muon=True,
                ns_steps=ns_steps,
            ),
            dict(
                params=aux_params,
                lr=args.learning_rate / 300,
                momentum=args.momentum,
                weight_decay=1e-4,
                use_muon=False,
            ),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    elif args.optimizer == "NorMuon":
        muon_params, aux_params = q_network.get_split_params()
        param_groups = [
            dict(params=muon_params, lr=args.learning_rate, weight_decay=1e-4, use_muon=True),
            dict(params=aux_params, lr=args.learning_rate / 300, weight_decay=1e-4, use_muon=False),
        ]
        optimizer = SingleDeviceNorMuonWithAuxAdam(param_groups)
    elif args.optimizer == "AdaMuon":
        muon_params, aux_params = q_network.get_split_params()
        param_groups = [
            dict(params=muon_params, lr=args.learning_rate, weight_decay=1e-4, use_muon=True),
            dict(params=aux_params, lr=args.learning_rate / 300, weight_decay=1e-4, use_muon=False),
        ]
        optimizer = AdaMuonWithAuxAdam(param_groups)
    elif args.optimizer == "BGD":
        params = BGD.create_unique_param_groups(q_network)
        optimizer = BGD(
            params,
            std_init=0.01,
            mean_eta=args.learning_rate,
            std_eta=10,
            betas=(args.momentum, 0.999, 0.99),
            mc_iters=1,
        )
        MC_Method = True
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    avg_returns = deque(maxlen=20)

    # preserve per-parameter-group LR ratios during annealing
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]

    # start
    global_step = 0
    start_time = time.time()
    next_obs = torch.as_tensor(envs.reset(), device=device)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.float32)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for g in optimizer.param_groups:
                g["lr"] = frac * g["initial_lr"]

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
                global_step,
            )

            random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,), device=device)

            with torch.no_grad():
                if MC_Method:
                    optimizer.randomize_weights(force_std=0)
                q_values = q_network(next_obs)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs, device=device), max_actions].flatten()

            explore = torch.rand((args.num_envs,), device=device) < epsilon
            action = torch.where(explore, random_actions, max_actions)
            actions[step] = action

            next_obs_np, reward, next_done_np, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.as_tensor(reward, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs_np, device=device)
            next_done = torch.as_tensor(next_done_np, device=device, dtype=torch.float32)

            for idx, d in enumerate(next_done_np):
                if d and info["lives"][idx] == 0:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # Compute Q(lambda) targets
        with torch.no_grad():
            returns = torch.zeros_like(rewards, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    if MC_Method:
                        optimizer.randomize_weights(force_std=0)
                    next_value, _ = torch.max(q_network(next_obs), dim=-1)
                    nextnonterminal = 1.0 - next_done.float()
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    next_value = values[t + 1]
                    returns[t] = (
                        rewards[t]
                        + args.gamma * (args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * next_value) * nextnonterminal
                    )

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)

        # optimize Q-network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                if MC_Method:
                    optimizer.randomize_weights()

                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                old_val = q_network(b_obs[mb_inds]).gather(1, b_actions[mb_inds].unsqueeze(-1).long()).squeeze()
                loss = F.mse_loss(b_returns[mb_inds], old_val)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                if MC_Method:
                    optimizer.aggregate_grads(1)
                optimizer.step()

        writer.add_scalar("losses/td_loss", loss.item(), global_step)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()

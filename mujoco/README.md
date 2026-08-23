# MuJoCo Actor-Critic PQN

This experiment adds continuous-action Q-learning to the repository.
It follows PQN's central recipe—short parallel online rollouts, network
normalization, and Q(lambda) regression without a replay buffer or target
network—and uses a deterministic DDPG-style actor to approximate
`argmax_a Q(s, a)`.

## Files and layout

- `cleanrl/pqn_mujoco.py`: CleanRL-style PyTorch trainer for any Gymnasium environment
  with flat observations and a bounded continuous action space.
- `mujoco/tune_mujoco.py`: optimizer-specific Optuna tuning over several environments
  and seeds, with one subprocess per GPU and per-environment score
  normalization.
- `mujoco/evaluate_mujoco.py`: resumable final evaluation using the JSON produced by
  the tuner. It writes per-run results, environment aggregates, overall
  normalized aggregates, and seed-paired optimizer deltas.

## Install

From the repository root:

```bash
pip install -e '.[mujoco,optuna]'
```

The repository currently pins Gymnasium 0.29 and MuJoCo 2.3, so the examples
use the `-v4` environment IDs.

## Train one run

Adam critic:

```bash
python cleanrl/pqn_mujoco.py \
  --env-id HalfCheetah-v4 \
  --optimizer Adam \
  --critic-learning-rate 3e-4 \
  --actor-learning-rate 3e-4 \
  --total-timesteps 1000000
```

Muon critic:

```bash
python cleanrl/pqn_mujoco.py \
  --env-id HalfCheetah-v4 \
  --optimizer Muon \
  --critic-learning-rate 3e-3 \
  --actor-learning-rate 3e-4 \
  --total-timesteps 1000000
```

By default the actor uses Adam in both conditions. This isolates the critic/Q
optimizer, which is the cleanest Adam-versus-Muon causal comparison. In the
Muon critic, every matrix parameter—including the scalar Q head's `1 x hidden`
matrix—uses Muon; biases and LayerNorm affine parameters use the optimizer's
auxiliary Adam path. Pass `--muon-actor` for a separate all-matrix Muon
ablation.

Every run writes `config.json`, TensorBoard events, `progress.csv`, and
`summary.json`. Pass `--save-model` to also save the actor, critic, and frozen
observation-normalization statistics.

## Tune each optimizer

Run separate studies because Adam and Muon generally need different learning
rates:

```bash
python mujoco/tune_mujoco.py \
  --optimizer Adam \
  --gpus 0,1 \
  --trials 30 \
  --seeds 2 \
  --study-name mujoco_pqn_adam

python mujoco/tune_mujoco.py \
  --optimizer Muon \
  --gpus 0,1 \
  --trials 30 \
  --seeds 2 \
  --study-name mujoco_pqn_muon
```

The default `core` search tunes critic learning rate, actor learning rate,
Q(lambda), and exploration-noise magnitude. `--search-space full` additionally
tunes rollout reuse, actor update frequency, exploration duration, and width.
The objective averages normalized deterministic evaluation return across
HalfCheetah, Hopper, and Walker2d. Override the normalization windows with
`--score-ranges-json ranges.json` when using other tasks.

Use `--dry-run` to print every command without launching training. Use
`--gpus cpu` for a CPU launch test.

## Evaluate tuned configurations

The evaluator consumes the tuner's `best_hyperparams.json` files directly:

```bash
python mujoco/evaluate_mujoco.py \
  --configs \
    mujoco/tuner_logs/mujoco_pqn_adam/best_hyperparams.json \
    mujoco/tuner_logs/mujoco_pqn_muon/best_hyperparams.json \
  --envs HalfCheetah-v4,Hopper-v4,Walker2d-v4,Ant-v4 \
  --gpus 0,1 \
  --num-seeds 10 \
  --total-timesteps 1000000 \
  --greedy-eval-steps 50000 \
  --greedy-eval-num-envs 8
```

After each training run, the evaluator executes the learned deterministic actor
for a fixed number of transitions with no exploration noise and no parameter
updates. In continuous action spaces this actor is the learned approximation to
`argmax_a Q(s, a)`. Only completed episodes enter the performance statistic;
the default 50,000 transitions across eight environments normally produces
dozens of MuJoCo episodes. This fixed-budget greedy return is the primary metric
in all aggregate and paired CSVs. The original fixed-number-of-episodes result
is retained as a secondary diagnostic.

Completed runs with the requested greedy evaluation budget are reused unless
`--overwrite` is set. Older summaries without the requested greedy evaluation
are rerun. The principal outputs are:

- `run_results.csv`: every seed, including failures and log locations;
- `environment_summary.csv`: raw and normalized return by game;
- `overall_summary.csv`: aggregate normalized return and throughput;
- `paired_deltas.csv`: same-game, same-seed contender-minus-Adam differences.

## Slurm/SuperPOD

Two Slurm files in `mujoco/` reproduce the tuning and final-evaluation workflow:

- `mujoco/mujoco_pqn_tune.sbatch` tunes Adam and Muon separately on all allocated GPUs;
- `mujoco/mujoco_pqn_evaluate.sbatch` loads both best-config JSON files and evaluates
  them with matched environments and seeds.

Submit the complete dependency-linked pipeline with:

```bash
bash mujoco/submit_mujoco_pqn_pipeline.sh
```

This submits both jobs immediately, but Slurm holds evaluation until tuning
finishes successfully. They can also be submitted independently:

```bash
sbatch mujoco/mujoco_pqn_tune.sbatch
sbatch mujoco/mujoco_pqn_evaluate.sbatch
```

The editable settings are grouped at the top of each file. The defaults request
eight GPUs in the `batch` partition using the same account and `RLMuon` Conda
environment as the repository's existing PQN Slurm jobs.

## Algorithm details

For a behavior transition `(s_t, a_t, r_t, s_{t+1})`, the actor supplies the
continuous greedy approximation `pi(s_{t+1})`, and the critic supplies
`Q(s_{t+1}, pi(s_{t+1}))`. Inside a rollout, targets are

```text
G_t^lambda = r_t + gamma * [
    (1 - lambda) Q(s_{t+1}, pi(s_{t+1})) + lambda G_{t+1}^lambda
].
```

Gaussian exploration is applied only to behavior actions. The critic regresses
the behavior-action values onto the Q(lambda) target; the actor maximizes the
current critic. Time-limit truncations cut the lambda trace but still
bootstrap, while true terminations do not bootstrap.

This is an online actor-critic Q-learning experiment, not SAC/TD3: there is no
replay buffer, target network, entropy term, or double critic. Those omissions
are deliberate so the implementation retains the instability and low-cost
online-update setting that makes PQN relevant to the optimizer question.

## References

- [Simplifying Deep Temporal Difference Learning](https://arxiv.org/abs/2407.04811)
- [PureJaxQL continuous-action PQN reference](https://github.com/mttga/purejaxql)

# DeepSea PQN: Adam versus Muon

This folder is the DeepSea version of the repository's PQN experiment stack.
Unlike the continuous MuJoCo adaptation, this is literal discrete Q-learning:
there is no actor and no approximation to the greedy action. The behavior
policy is epsilon-greedy over `argmax_a Q(s, a)` and the learner uses the same
parallel online Q(lambda) regression structure as `cleanrl/pqn.py` and
`cleanrl/pqn_atari_envpool.py`—with no replay buffer and no target network.

## What is implemented

`cleanrl/pqn_deepsea.py` contains both the trainer and a GPU-vectorized,
bsuite-style DeepSea environment. An `N x N` one-hot state starts at the top
left. Every state has a random, fixed mapping from the two action labels to
left/right. Moving right costs `0.01/N`, and reaching the lower-right treasure
adds `+1`. An episode uses the conventional `N-1` downward transitions from
the top to the bottom row. All parallel
environments in a run share
one action map, while a different run seed generates a different map.

The trainer avoids storing large one-hot rollout tensors: it stores integer
state IDs and expands them to `N^2` one-hot inputs only during each forward
pass. The MLP uses LayerNorm and ReLU by default. With Muon, the input and
hidden matrix weights use Muon while biases, normalization parameters, and the
Q head use the auxiliary Adam path, matching this repository's Atari routing.
Use `--use-muon-output` to add the Q-head matrix to Muon as an explicit
all-matrix ablation.

The root-level experiment tools are:

- `deepsea/tune_deepsea.py`: separate optimizer-specific Optuna studies over
  multiple sizes and seeds, distributed as one subprocess per GPU;
- `deepsea/evaluate_deepsea.py`: resumable, matched-seed final evaluation that
  consumes the tuner's JSON directly and writes per-run, per-size, overall,
  and paired Adam-versus-Muon tables;
- `deepsea/deepsea_pqn_tune.sbatch` and
  `deepsea/deepsea_pqn_evaluate.sbatch`: eight-GPU Slurm jobs;
- `deepsea/submit_deepsea_pqn_pipeline.sh`: submits evaluation with an
  `afterok` dependency on tuning.

## Install

No external environment package is required. From the repository root:

```bash
pip install -e '.[optuna]'
```

## Train one run

```bash
python cleanrl/pqn_deepsea.py \
  --deepsea-size 20 \
  --optimizer Adam \
  --learning-rate 3e-4 \
  --total-timesteps 1000000

python cleanrl/pqn_deepsea.py \
  --deepsea-size 20 \
  --optimizer Muon \
  --learning-rate 3e-3 \
  --total-timesteps 1000000
```

W&B tracking is enabled by default. Pass `--no-track` to disable it. Every run
also writes local `config.json`, TensorBoard events, `progress.csv`, and
`summary.json` under `logs/deepsea/runs/` unless `--output-dir` is supplied.
The trainer sends metrics directly to W&B rather than relying on TensorBoard
sync. Important keys include training return/success, TD loss, gradient norm,
SPS, greedy evaluation return/success, and nonterminal-grid action accuracy.

## Tune Adam and Muon separately

```bash
python deepsea/tune_deepsea.py \
  --optimizer Adam \
  --sizes 10,20,30 \
  --gpus 0,1 \
  --trials 30 \
  --seeds 3 \
  --study-name deepsea_pqn_adam

python deepsea/tune_deepsea.py \
  --optimizer Muon \
  --sizes 10,20,30 \
  --gpus 0,1 \
  --trials 30 \
  --seeds 3 \
  --study-name deepsea_pqn_muon
```

The exploration fraction is fixed at `0.1`: epsilon always anneals over the
first 10% of training. The default `core` study tunes learning rate, Q(lambda),
and the minimum/final epsilon over the deliberately constrained set
`{0, 0.001, 0.003, 0.01, 0.03, 0.05}`. Larger final epsilons are omitted
because persistent independent action noise rapidly destroys coherent
right-moving trajectories as DeepSea grows. `--search-space full` additionally
tunes update epochs and hidden width. The objective is mean normalized greedy
return over all requested size/seed runs. Return normalization uses the
conservative lower bound `-0.01` and the size-specific optimal return
`1 - 0.01(N-1)/N`, so an optimal run maps to one.

Tuning outputs live under `logs/deepsea/tuning/<study-name>/`, including the
Optuna database, subprocess logs, every run's full local metrics, and
`best_hyperparams.json`. Use `--dry-run` to inspect commands or `--no-track` to
disable W&B for the spawned trainers.

## Final matched-seed evaluation

```bash
python deepsea/evaluate_deepsea.py \
  --configs \
    logs/deepsea/tuning/deepsea_pqn_adam/best_hyperparams.json \
    logs/deepsea/tuning/deepsea_pqn_muon/best_hyperparams.json \
  --sizes 10,20,30,40,50 \
  --gpus 0,1 \
  --num-seeds 10 \
  --total-timesteps 1000000
```

After training, each run executes a separate epsilon-zero episode batch on the
same fixed map. The evaluator reports greedy return and treasure success as the
primary performance measures, plus the fraction of all grid states at which
the learned greedy action matches the map's right action (excluding the
terminal row). Existing runs are
reused only when their saved configuration exactly matches the current request;
pass `--overwrite` to force reruns.

The final tables are written to `logs/deepsea/evaluation/`:

- `run_results.csv`
- `size_summary.csv`
- `overall_summary.csv`
- `paired_deltas.csv`

## Slurm pipeline

From the repository root:

```bash
bash deepsea/submit_deepsea_pqn_pipeline.sh
```

The submission helper creates `logs/deepsea/slurm/` before Slurm opens its
output files, submits tuning, and makes evaluation depend on successful tuning.
To submit either job independently, first create that directory:

```bash
mkdir -p logs/deepsea/slurm
sbatch deepsea/deepsea_pqn_tune.sbatch
sbatch deepsea/deepsea_pqn_evaluate.sbatch
```

W&B is enabled in both Slurm files (`TRACK=1`) and can be disabled in their
editable settings blocks.

## Scientific caveat

DeepSea was designed to expose *deep exploration*. Plain stepwise
epsilon-greedy exploration reaches the treasure with probability roughly
`2^-N` before it has learned a useful path. Therefore, failure at `N=30–50` is
an expected limitation of this behavior policy, not automatically evidence of
optimizer instability. This package cleanly asks whether Muon's update geometry
changes learning or robustness under that fixed exploration mechanism; it does
not silently add bootstrapping ensembles, parameter noise, or an exploration
bonus. If both optimizers fail on large sizes, report the scaling boundary and
do not claim that Muon supplies deep exploration by itself.

## References

- [Simplifying Deep Temporal Difference Learning](https://arxiv.org/abs/2407.04811)
- [Behaviour Suite for Reinforcement Learning](https://arxiv.org/abs/1908.03568)
- [bsuite reference repository](https://github.com/google-deepmind/bsuite)

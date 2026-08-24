# Craftax symbolic PQN: Adam versus Muon

This is the Craftax counterpart of the repository's PyTorch PQN experiments.
It is literal discrete Q-learning: the behavior policy is epsilon-greedy over
`argmax_a Q(s,a)`, training uses replay-free Q(lambda) rollout targets, and
there is no actor, target network, or continuous-action approximation.

The default environment is `Craftax-Classic-Symbolic-v1`, Craftax's JAX
reimplementation of the original Crafter mechanics with symbolic observations.
The same files support the larger `Craftax-Symbolic-v1` through `--env-id` or
`--envs`. The full environment is a substantially harder benchmark and its
partially observed dynamics may eventually justify a recurrent model; this
package deliberately starts with the feed-forward comparison used elsewhere
in the repository.

## How JAX fits into the PyTorch repository

`cleanrl/pqn_craftax.py` keeps the Q-network, loss, Q(lambda) calculation,
Adam/Muon optimizers, TensorBoard, CSV, and W&B logging in PyTorch. Only Craftax
environment reset and step live in JAX. The environments are batched with
`jax.vmap` and compiled with `jax.jit`; observations, actions, rewards, and done
flags cross frameworks using DLPack. On one GPU this normally avoids a host
copy. `XLA_PYTHON_CLIENT_PREALLOCATE=false` and a default JAX memory fraction of
`0.35` prevent JAX from reserving the entire accelerator before PyTorch starts.

This boundary does impose one PyTorch-to-JAX and one JAX-to-PyTorch transition
per vector environment step, so it will not match the throughput of a fully
JAX `lax.scan` learner. It keeps the optimizer comparison faithful to the
existing PyTorch code, which is the important constraint here.

## Files

- `cleanrl/pqn_craftax.py`: single-run trainer and final greedy evaluation.
- `craftax_pqn/tune_craftax.py`: separate Adam and Muon Optuna studies, one
  subprocess per GPU.
- `craftax_pqn/evaluate_craftax.py`: matched-seed, resumable final evaluation
  that consumes the tuning JSON files directly.
- `craftax_pqn/*.sbatch`: eight-GPU tuning and evaluation jobs.
- `craftax_pqn/submit_craftax_pqn_pipeline.sh`: dependency-aware Slurm submitter.

## Install

Install the repository and tuning extras, then Craftax and a JAX build matching
the machine. For a CUDA 12 environment, for example:

```bash
pip install -e '.[optuna]'
pip install craftax
pip install --upgrade 'jax[cuda12]'
```

For CPU-only command checks, use `pip install --upgrade jax` instead. Confirm
that both frameworks see the intended accelerator before launching a study:

```bash
python -c 'import torch, jax, craftax; print(torch.cuda.is_available(), jax.devices())'
```

## One training run

```bash
python cleanrl/pqn_craftax.py \
  --env-id Craftax-Classic-Symbolic-v1 \
  --optimizer Adam \
  --learning-rate 3e-4 \
  --total-timesteps 1000000

python cleanrl/pqn_craftax.py \
  --env-id Craftax-Classic-Symbolic-v1 \
  --optimizer Muon \
  --learning-rate 3e-3 \
  --total-timesteps 1000000
```

W&B is enabled by default and receives metrics directly. Every run also writes
`config.json`, TensorBoard events, `progress.csv`, and `summary.json` under
`logs/craftax/runs/` unless `--output-dir` is set. The important metrics are
training episode return/length, TD loss, Q values, gradient norm, SPS, and the
separate post-training epsilon-zero return. Pass `--no-track` to disable W&B.

Muon routing matches the repository's Atari convention: the input matrix and
all hidden matrices use Muon, while biases, LayerNorm parameters, and the Q
head use auxiliary Adam. `--use-muon-output` adds the Q head as an explicit
all-matrix ablation.

## Tuning

```bash
python craftax_pqn/tune_craftax.py \
  --optimizer Adam \
  --envs Craftax-Classic-Symbolic-v1 \
  --gpus 0,1 \
  --trials 30 \
  --seeds 3 \
  --study-name craftax_pqn_adam

python craftax_pqn/tune_craftax.py \
  --optimizer Muon \
  --envs Craftax-Classic-Symbolic-v1 \
  --gpus 0,1 \
  --trials 30 \
  --seeds 3 \
  --study-name craftax_pqn_muon
```

The tuner uses the requested setup exactly:

- exploration fraction fixed at `0.1`;
- final epsilon in `{0, .001, .003, .01, .03, .05}`;
- `distance_from_one ~ Uniform(0, .5)` and
  `q_lambda = 1 - distance_from_one^2`;
- Muon learning rate log-uniform from `3e-4` to `3e-2`;
- Adam learning rate log-uniform from `3e-5` to `3e-3`;
- two LayerNorm hidden layers, input Muon on, output Muon off;
- core search fixes four update epochs and width 256; full search adds epochs
  in `{1,2,4}` and width in `{128,256,512}`.

The objective is post-training greedy return, normalized by the number of
achievements (22 for Classic and 226 for full Craftax) and averaged over
matched environment/seed runs. Normalized scores are not clipped. The saved
`best_hyperparams.json` reproduces the squared lambda transform used during the
trial.

Use `--dry-run` to inspect all spawned commands without importing JAX, Craftax,
Torch, or Optuna inside the training subprocess.

## Final matched-seed evaluation

```bash
python craftax_pqn/evaluate_craftax.py \
  --configs \
    logs/craftax/tuning/craftax_pqn_adam/best_hyperparams.json \
    logs/craftax/tuning/craftax_pqn_muon/best_hyperparams.json \
  --envs Craftax-Classic-Symbolic-v1 \
  --gpus 0,1 \
  --num-seeds 10 \
  --total-timesteps 1000000
```

After every training run the trainer creates fresh procedurally generated
evaluation environments and executes a fixed number of epsilon-zero steps.
The evaluator reports raw and normalized completed-episode return, reward per
1,000 evaluation steps, throughput, and matched Adam-to-Muon deltas. Existing
runs are reused only when their saved configuration exactly matches the
request; use `--overwrite` to force a rerun.

Tables are written to `logs/craftax/evaluation/`:

- `run_results.csv`
- `environment_summary.csv`
- `overall_summary.csv`
- `paired_deltas.csv`

## Slurm

```bash
bash craftax_pqn/submit_craftax_pqn_pipeline.sh
```

The defaults request eight GPUs and run optimizer studies sequentially, giving
each study all eight GPUs. The evaluator starts only if tuning succeeds. W&B is
enabled in both job files (`TRACK=1`). Review the account, partition, conda
environment, wall time, and JAX CUDA installation for the target cluster.

## Interpretation caveats

One million environment steps is appropriate for a fast optimizer screen, not
a definitive Craftax result. Any paper claim should rerun the selected
configurations at a larger budget and with matched seeds. Also compare on raw
return and reward rate, not only achievement-count normalization. If the full
Craftax feed-forward runs remain weak, add a recurrent experiment as a separate
architectural study rather than changing only one optimizer's model.

## References

- [Craftax official repository](https://github.com/MichaelTMatthews/Craftax)
- [PureJaxRL/PureJaxQL Craftax baselines](https://github.com/mttga/purejaxql)
- [Simplifying Deep Temporal Difference Learning](https://arxiv.org/abs/2407.04811)

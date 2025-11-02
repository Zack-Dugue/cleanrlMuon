"""
atari10_tune_envpool.py
- Tunes a CleanRL Atari agent implemented with EnvPool across a 10-game subset.
- Uses MultiGPUTuner (your fail-fast, spawn-safe tuner).
- Passes a fixed --optimizer STRING down to the training script (not tuned).
- Normalizes episodic returns with rough per-game ranges (adjust as you like).

Example:
  python atari10_tune_envpool.py \
    --script cleanrl/ppo_atari_envpool.py \
    --gpus 0,1 \
    --optimizer adamw \
    --trials 40 \
    --seeds 3 \
    --study-name atari10_envpool_adamw

Notes:
- Make sure `cleanrl/ppo_atari_envpool.py` is your EnvPool-based script that accepts:
    --env-id (set by the tuner)
    --optimizer (this script passes it through)
    --learning-rate, --ent-coef, --update-epochs, --momentum, --num-envs, --num-steps, --total-timesteps
- The metric read is "charts/episodic_return" from TensorBoard logs under runs/<run_name>.
"""

from __future__ import annotations
import argparse
from typing import Dict, Optional, List

import optuna

from cleanrl_utils.multigpu_tuner import MultiGPUTuner


# ---------- 10-game subset (Gymnasium Atari v5 ids) ----------
ATARI10: List[str] = [
    "Pong-v5",
    "Breakout-v5",
    "Freeway-v5",
    "Enduro-v5",
    "Seaquest-v5",
    "SpaceInvaders-v5",
    "MsPacman-v5",
    "MontezumaRevenge-v5",
    "Assault-v5",
    "Gravitar-v5",
]

# Rough normalization windows (min, max) for episodic return.
# These are pragmatic defaults to stabilize cross-game aggregation; feel free to refine.
TARGET_SCORES: Dict[str, Optional[List[float]]] = {
    "Pong-v5":              [-21.0, 21.0],
    "Breakout-v5":          [0.0, 400.0],
    "Freeway-v5":           [0.0, 34.0],
    "Enduro-v5":            [0.0, 3000.0],
    "Seaquest-v5":          [0.0, 50000.0],
    "SpaceInvaders-v5":     [0.0, 2000.0],
    "MsPacman-v5":          [0.0, 15000.0],
    "MontezumaRevenge-v5":  [0.0, 10000.0],
    "Assault-v5":           [0.0, 5000.0],
    "Gravitar-v5":          [0.0, 1000.0],
}


def default_params_fn(optimizer_name: str):
    """
    Closure that captures the fixed optimizer string and returns an Optuna params_fn.
    The tuner will inject --env-id and --seed for each run.
    """
    def _fn(trial: optuna.Trial) -> dict:
        return {
            # Tunables (kebab-case flags as expected by your CleanRL script)
            "learning-rate": trial.suggest_float("lr", 3e-5, 3e-3, log=True),
            "ent-coef": trial.suggest_float("ent_coef", 0.0, 0.02),
            "update-epochs": trial.suggest_int("update_epochs", 2, 8),
            "momentum": trial.suggest_float("momentum", 0.7, 0.99),

            # Fixed batch shape / run length
            "num-envs": 32,
            "num-steps": 256,               # total batch: 8192
            "total-timesteps": 5_000_000,

            # Fixed (not tuned) — passed through to the training script
            "optimizer": optimizer_name,

            # If your script needs an explicit switch for EnvPool, add it here, e.g.:
            # "env-backend": "envpool",  # Only if your script expects this flag
        }
    return _fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--script", type=str, default="cleanrl/ppo_atari_envpool.py",
                   help="Path to the EnvPool-based CleanRL script.")
    p.add_argument("--metric", type=str, default="charts/episodic_return",
                   help="TensorBoard scalar to read.")
    p.add_argument("--gpus", type=str, default="0",
                   help="Comma-separated CUDA device indices local to this node, e.g. '0,1'.")
    p.add_argument("--optimizer", type=str, default="adamw",
                   help="Optimizer string to pass through to the training script (not tuned).")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--study-name", type=str, default="atari10_envpool_study")
    p.add_argument("--storage", type=str, default="sqlite:///cleanrl_hpopt.db",
                   help="Optuna storage (use MySQL/Postgres for multi-node).")
    p.add_argument("--logs-root", type=str, default="tuner_logs")
    p.add_argument("--metric-window", type=int, default=50,
                   help="Average last N scalars from TB for the metric.")
    args = p.parse_args()

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]

    tuner = MultiGPUTuner(
        script=args.script,
        metric=args.metric,
        gpus=gpu_list,
        target_scores={gid: TARGET_SCORES[gid] for gid in ATARI10},
        params_fn=default_params_fn(args.optimizer),
        direction="maximize",
        aggregation_type="average",
        metric_last_n_average_window=args.metric_window,
        sampler=optuna.samplers.TPESampler(seed=123),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        storage=args.storage,
        study_name=args.study_name,
        wandb_kwargs={},  # fill to enable Weights & Biases
        logs_root=args.logs_root,
    )

    best = tuner.tune(num_trials=args.trials, num_seeds=args.seeds)
    print("Best:", best.value, best.params)


if __name__ == "__main__":
    main()

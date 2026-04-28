"""
atari10_tune_envpool.py
- Tunes a CleanRL Atari agent implemented with EnvPool across a 10-game subset.
- Uses MultiGPUTuner (your fail-fast, spawn-safe tuner).
- Passes a fixed --optimizer STRING down to the training script (not tuned).
- Normalizes episodic returns with rough per-game ranges (adjust as you like).
- Saves the best trial to text/JSON files under <logs_root>/<study_name>/.
"""

from __future__ import annotations
import argparse
import json
import os
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
    "Assault-v5",
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
    "Assault-v5":           [0.0, 5000.0],
}


def default_params_fn(optimizer_name: str):
    """
    Closure that captures the fixed optimizer string and returns an Optuna params_fn.
    The tuner will inject --env-id and --seed for each run.
    """
    def _fn(trial: optuna.Trial) -> dict:
        if optimizer_name in ["Muon", "NorMuon", "AdaMuon"]:
            learning_rate = trial.suggest_float("lr", 3e-4, 3e-2, log=True)
        else:
            learning_rate = trial.suggest_float("lr", 3e-5, 3e-3, log=True)

        return {
            # Tunables
            "learning-rate": learning_rate,
            "momentum": trial.suggest_float("momentum", 0.9, 0.99),
            "exploration-fraction": trial.suggest_float("exploration_fraction", 0.03, 0.20),
            "q-lambda": trial.suggest_float("q_lambda", 0.45, 0.85),

            # Fixed batch shape / run length
            "num-envs": 32,
            "num-steps": 256,               # total batch: 8192
            "total-timesteps": 5_000_000,

            # Fixed (not tuned)
            "optimizer": optimizer_name,
        }
    return _fn


def save_best_trial(best, study_name: str, logs_root: str, optimizer_name: str, script: str, metric: str,
                    trials: int, seeds: int, gpu_list: List[int]):
    study_dir = os.path.join(logs_root, study_name)
    os.makedirs(study_dir, exist_ok=True)

    txt_path = os.path.join(study_dir, "best_hyperparams.txt")
    json_path = os.path.join(study_dir, "best_hyperparams.json")

    payload = {
        "study_name": study_name,
        "optimizer": optimizer_name,
        "script": script,
        "metric": metric,
        "trials_requested": int(trials),
        "seeds_per_trial": int(seeds),
        "gpus": list(gpu_list),
        "best_value": float(best.value),
        "best_trial_number": int(best.number),
        "best_params": dict(best.params),
    }

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"study_name: {payload['study_name']}\n")
        f.write(f"optimizer: {payload['optimizer']}\n")
        f.write(f"script: {payload['script']}\n")
        f.write(f"metric: {payload['metric']}\n")
        f.write(f"trials_requested: {payload['trials_requested']}\n")
        f.write(f"seeds_per_trial: {payload['seeds_per_trial']}\n")
        f.write(f"gpus: {payload['gpus']}\n")
        f.write(f"best_trial_number: {payload['best_trial_number']}\n")
        f.write(f"best_value: {payload['best_value']}\n")
        f.write("best_params:\n")
        for k, v in payload["best_params"].items():
            f.write(f"  {k}: {v}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return txt_path, json_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--script", type=str, default="cleanrl/pqn_atari_envpool.py",
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
    p.add_argument("--wandb-tag", type=str, default=None,
                   help="Optional WandB tag for grouping/filtering runs.")
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
        wandb_tag=args.wandb_tag,
    )

    best = tuner.tune(num_trials=args.trials, num_seeds=args.seeds)
    txt_path, json_path = save_best_trial(
        best=best,
        study_name=args.study_name,
        logs_root=args.logs_root,
        optimizer_name=args.optimizer,
        script=args.script,
        metric=args.metric,
        trials=args.trials,
        seeds=args.seeds,
        gpu_list=gpu_list,
    )
    print("Best:", best.value, best.params)
    print(f"Saved best hyperparameters to: {txt_path}")
    print(f"Saved best hyperparameters JSON to: {json_path}")


if __name__ == "__main__":
    main()

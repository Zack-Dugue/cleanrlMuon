"""
atari10_tune_envpool.py
- Tunes a CleanRL Atari agent implemented with EnvPool across a 10-game subset.
- Uses MultiGPUTuner.
- Passes a fixed --optimizer STRING down to the training script.
- Normalizes episodic returns with rough per-game ranges.

Example:
  python atari10_tune_envpool.py \
    --script cleanrl/ppo_atari_envpool.py \
    --gpus 0,1 \
    --optimizer adamw \
    --trials 40 \
    --seeds 3 \
    --study-name atari10_envpool_adamw \
    --wandb-tag muon_input

Notes:
- Make sure `cleanrl/ppo_atari_envpool.py` accepts:
    --env-id
    --seed
    --optimizer
    --learning-rate
    --ent-coef
    --update-epochs
    --momentum
    --num-envs
    --num-steps
    --total-timesteps
- The metric read is "charts/episodic_return" from TensorBoard logs under runs/<run_name>.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

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
    "Pong-v5": [-21.0, 21.0],
    "Breakout-v5": [0.0, 400.0],
    "Freeway-v5": [0.0, 34.0],
    "Enduro-v5": [0.0, 3000.0],
    "Seaquest-v5": [0.0, 50000.0],
    "SpaceInvaders-v5": [0.0, 2000.0],
    "MsPacman-v5": [0.0, 15000.0],
    "Assault-v5": [0.0, 5000.0],
}


def _safe_slug(x) -> str:
    """
    Make a string safe for filenames.
    """
    if x is None:
        return "none"

    x = str(x).strip()
    if not x:
        return "none"

    x = re.sub(r"[^A-Za-z0-9_.=-]+", "_", x)
    return x[:120]


def default_params_fn(optimizer_name: str):
    """
    Closure that captures the fixed optimizer string and returns an Optuna params_fn.
    The tuner will inject --env-id and --seed for each run.
    """

    def _fn(trial: optuna.Trial) -> dict:
        return {
            # Tunables, kebab-case flags as expected by your CleanRL script.
            "learning-rate": trial.suggest_float("lr", 3e-5, 3e-3, log=True),
            "ent-coef": trial.suggest_float("ent_coef", 0.001, 0.03),
            "clip-coef": trial.suggest_float("clip_coef", 0.05, .3),
            "update-epochs": trial.suggest_int("update_epochs", 5, 5),
            "momentum": trial.suggest_float("momentum", 0.95, 0.95),

            # Fixed batch shape / run length.
            "num-envs": 32,
            "num-steps": 256,  # total batch: 8192
            "total-timesteps": 5_000_000,

            # Fixed optimizer choice.
            "optimizer": optimizer_name,
        }

    return _fn


def write_optuna_summary_file(
    *,
    args,
    best: optuna.trial.FrozenTrial,
    target_scores: Dict[str, Optional[List[float]]],
) -> Path:
    """
    Write a detailed Optuna summary report once, after tuning finishes.

    This does not run during training, so it will not affect PPO speed.
    Filename is intentionally short:
      <study_name>__<wandb_tag>.txt
    """
    results_dir = Path(args.logs_root) / args.study_name / "optuna_summaries"
    results_dir.mkdir(parents=True, exist_ok=True)

    study_slug = _safe_slug(args.study_name)
    tag_slug = _safe_slug(args.wandb_tag)

    out_path = results_dir / f"{study_slug}__{tag_slug}.txt"

    # Avoid overwriting an existing summary from an earlier run with the same study/tag.
    if out_path.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"{study_slug}__{tag_slug}__{timestamp}.txt"

    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage,
    )

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]
    failed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.FAIL
    ]

    reverse = study.direction == optuna.study.StudyDirection.MAXIMIZE
    completed_sorted = sorted(
        completed_trials,
        key=lambda t: float("-inf") if t.value is None else t.value,
        reverse=reverse,
    )

    args_dict = vars(args)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("OPTUNA HPO SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        f.write("FINAL BEST RESULT\n")
        f.write("-" * 100 + "\n")
        f.write(f"best_trial_number: {best.number}\n")
        f.write(f"best_value:        {best.value}\n")
        f.write(f"direction:         {study.direction.name}\n")
        f.write(f"study_name:        {args.study_name}\n")
        f.write(f"wandb_tag:         {args.wandb_tag}\n")
        f.write(f"optimizer:         {args.optimizer}\n")
        f.write(f"metric:            {args.metric}\n")
        f.write(f"metric_window:     {args.metric_window}\n")
        f.write("\n")

        f.write("BEST HYPERPARAMETERS / OPTIMAL VARIABLES\n")
        f.write("-" * 100 + "\n")
        if best.params:
            for k, v in best.params.items():
                f.write(f"{k}: {v}\n")
        else:
            f.write("(none)\n")
        f.write("\n")

        f.write("BEST TRIAL INTERMEDIATE VALUES\n")
        f.write("-" * 100 + "\n")
        if best.intermediate_values:
            for step, value in best.intermediate_values.items():
                f.write(f"step_{step}: {value}\n")
        else:
            f.write("(none)\n")
        f.write("\n")

        f.write("RUN CONFIGURATION ARGS\n")
        f.write("-" * 100 + "\n")
        for k in sorted(args_dict.keys()):
            f.write(f"{k}: {args_dict[k]}\n")
        f.write("\n")

        f.write("TARGET SCORE NORMALIZATION WINDOWS\n")
        f.write("-" * 100 + "\n")
        for env_id, window in target_scores.items():
            f.write(f"{env_id}: {window}\n")
        f.write("\n")

        f.write("STUDY OVERVIEW\n")
        f.write("-" * 100 + "\n")
        f.write(f"total_trials_in_storage: {len(study.trials)}\n")
        f.write(f"completed_trials:        {len(completed_trials)}\n")
        f.write(f"pruned_trials:           {len(pruned_trials)}\n")
        f.write(f"failed_trials:           {len(failed_trials)}\n")
        f.write(f"sampler:                 {study.sampler.__class__.__name__}\n")
        f.write(f"pruner:                  {study.pruner.__class__.__name__}\n")
        f.write(f"storage:                 {args.storage}\n")
        f.write(f"script:                  {args.script}\n")
        f.write(f"gpus:                    {args.gpus}\n")
        f.write("\n")

        f.write("ALL COMPLETED TRIALS, BEST FIRST\n")
        f.write("-" * 100 + "\n")
        for rank, t in enumerate(completed_sorted, start=1):
            f.write(f"\nRANK {rank}\n")
            f.write(f"trial_number:      {t.number}\n")
            f.write(f"value:             {t.value}\n")
            f.write(f"state:             {t.state.name}\n")
            f.write(f"datetime_start:    {t.datetime_start}\n")
            f.write(f"datetime_complete: {t.datetime_complete}\n")

            f.write("params:\n")
            if t.params:
                for k, v in t.params.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write("  (none)\n")

            f.write("intermediate_values:\n")
            if t.intermediate_values:
                for step, value in t.intermediate_values.items():
                    f.write(f"  step_{step}: {value}\n")
            else:
                f.write("  (none)\n")

            f.write("user_attrs:\n")
            if t.user_attrs:
                for k, v in t.user_attrs.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write("  (none)\n")

        f.write("\n")
        f.write("ALL TRIALS RAW JSON SUMMARY\n")
        f.write("-" * 100 + "\n")

        raw_trials = []
        for t in study.trials:
            raw_trials.append(
                {
                    "number": t.number,
                    "state": t.state.name,
                    "value": t.value,
                    "params": t.params,
                    "intermediate_values": dict(t.intermediate_values),
                    "user_attrs": t.user_attrs,
                    "system_attrs": {k: str(v) for k, v in t.system_attrs.items()},
                    "datetime_start": str(t.datetime_start),
                    "datetime_complete": str(t.datetime_complete),
                }
            )

        f.write(json.dumps(raw_trials, indent=2, sort_keys=True))
        f.write("\n")

    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--script",
        type=str,
        default="cleanrl/ppo_atari_envpool.py",
        help="Path to the EnvPool-based CleanRL script.",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="charts/episodic_return",
        help="TensorBoard scalar to read.",
    )
    p.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated CUDA device indices local to this node, e.g. '0,1'.",
    )
    p.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="Optimizer string to pass through to the training script.",
    )
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--study-name", type=str, default="atari10_envpool_study")
    p.add_argument(
        "--storage",
        type=str,
        default="sqlite:///cleanrl_hpopt.db",
        help="Optuna storage. Use MySQL/Postgres for multi-node.",
    )
    p.add_argument("--logs-root", type=str, default="tuner_logs")
    p.add_argument(
        "--metric-window",
        type=int,
        default=50,
        help="Average last N scalars from TensorBoard for the metric.",
    )
    p.add_argument(
        "--wandb-tag",
        type=str,
        default=None,
        help="Optional W&B tag for grouping/filtering runs.",
    )

    args = p.parse_args()

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    target_scores = {gid: TARGET_SCORES[gid] for gid in ATARI10}

    tuner = MultiGPUTuner(
        script=args.script,
        metric=args.metric,
        gpus=gpu_list,
        target_scores=target_scores,
        params_fn=default_params_fn(args.optimizer),
        direction="maximize",
        aggregation_type="average",
        metric_last_n_average_window=args.metric_window,
        sampler=optuna.samplers.TPESampler(seed=123),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        storage=args.storage,
        study_name=args.study_name,
        wandb_kwargs={},
        logs_root=args.logs_root,
        wandb_tag=args.wandb_tag,
    )

    best = tuner.tune(num_trials=args.trials, num_seeds=args.seeds)

    summary_path = write_optuna_summary_file(
        args=args,
        best=best,
        target_scores=target_scores,
    )

    print(f"[done] Optuna summary written to: {summary_path}")
    print(f"[done] Best value: {best.value}")
    print(f"[done] Best params: {best.params}")


if __name__ == "__main__":
    main()
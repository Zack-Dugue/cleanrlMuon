"""
atari10_tune_corr_ablation.py

Self-contained Optuna tuner for the correlation-weighted PPO ablation.

It tunes exactly:
  - learning-rate
  - clip-coef
  - ent-coef

It passes Tyro-compatible boolean flags through to the training script:
runs a multi-game Atari subset, normalizes returns, and writes the best result
to JSON for later evaluation.

Designed to be launched twice by the SBATCH script:
  1. weighted/fancy version first
  2. unweighted baseline second
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
from tensorboard.backend.event_processing import event_accumulator


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


def str2bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    x = x.lower().strip()
    if x in {"true", "1", "yes", "y", "on"}:
        return True
    if x in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Could not parse boolean value: {x}")


def safe_slug(x: object) -> str:
    x = "none" if x is None else str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_.=-]+", "_", x)
    return x[:160] or "none"


def trial_params(trial: optuna.Trial) -> dict:
    return {
        "learning-rate": trial.suggest_float("lr", 3e-5, 3e-3, log=True),
        "clip-coef": trial.suggest_float("clip_coef", 0.05, 0.30),
        "ent-coef": trial.suggest_float("ent_coef", 0.001, 0.03),
    }


def find_new_run_dir(runs_dir: Path, before: set[Path], started_at: float) -> Path:
    after = set(p for p in runs_dir.glob("*") if p.is_dir())
    candidates = list(after - before)
    if not candidates:
        # Fallback: use modification time if directory set was already visible.
        candidates = [
            p for p in after
            if p.stat().st_mtime >= started_at - 2.0
        ]
    if not candidates:
        raise RuntimeError("Could not identify the run directory created by the training script.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_metric_from_tb(run_dir: Path, metric: str, window: int) -> float:
    ea = event_accumulator.EventAccumulator(str(run_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if metric not in tags:
        raise RuntimeError(f"Metric {metric!r} not found in {run_dir}. Available scalar tags include: {tags[:25]}")
    values = [e.value for e in ea.Scalars(metric)]
    if not values:
        raise RuntimeError(f"Metric {metric!r} exists but has no scalar values in {run_dir}.")
    return float(np.mean(values[-window:]))


def normalize_score(env_id: str, score: float) -> float:
    window = TARGET_SCORES[env_id]
    if window is None:
        return score
    lo, hi = window
    return (score - lo) / (hi - lo)


def run_one(
    *,
    args,
    gpu: int,
    env_id: str,
    seed: int,
    params: dict,
    trial_number: int,
) -> Tuple[str, int, float, float, str]:
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    before = set(p for p in runs_dir.glob("*") if p.is_dir())
    started_at = time.time()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    exp_name = (
        f"{safe_slug(args.exp_name)}"
        f"__trial{trial_number}"
        f"__{safe_slug('weighted' if args.use_correlation_weighting else 'unweighted')}"
    )

    cmd = [
        sys.executable,
        args.script,
        f"--env-id={env_id}",
        f"--seed={seed}",
        f"--optimizer={args.optimizer}",
        f"--learning-rate={params['learning-rate']}",
        f"--clip-coef={params['clip-coef']}",
        f"--ent-coef={params['ent-coef']}",
        f"--update-epochs={args.update_epochs}",
        f"--num-envs={args.num_envs}",
        f"--num-steps={args.num_steps}",
        f"--num-minibatches={args.num_minibatches}",
        f"--total-timesteps={args.total_timesteps}",
        f"--exp-name={exp_name}",
    ]

    # Tyro boolean flags should be passed by presence/negation, not as =True/=False.
    # With use_correlation_weighting defaulting to True in the training script, this is
    # still explicit for both ablation branches.
    if args.use_correlation_weighting:
        cmd.append("--use-correlation-weighting")
    else:
        cmd.append("--no-use-correlation-weighting")

    if args.aux_learning_rate is not None:
        cmd.append(f"--aux-learning-rate={args.aux_learning_rate}")
    if args.momentum is not None:
        cmd.append(f"--momentum={args.momentum}")
    if args.track_runs:
        cmd.append("--track")
        cmd.append(f"--wandb-project-name={args.wandb_project_name}")
        if args.wandb_entity:
            cmd.append(f"--wandb-entity={args.wandb_entity}")
        tag = args.wandb_tag or ("corr_weighted" if args.use_correlation_weighting else "corr_unweighted")
        cmd.append(f"--wandb-tag={tag}")

    log_dir = Path(args.logs_root) / args.study_name / "trial_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"trial{trial_number:04d}__seed{seed}__{safe_slug(env_id)}.log"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)

    if proc.returncode != 0:
        raise RuntimeError(f"Training failed for env={env_id}, seed={seed}, gpu={gpu}. See {log_path}")

    run_dir = find_new_run_dir(runs_dir, before, started_at)
    raw_score = read_metric_from_tb(run_dir, args.metric, args.metric_window)
    norm_score = normalize_score(env_id, raw_score)

    return env_id, seed, raw_score, norm_score, str(run_dir)


def write_best_artifacts(args, study: optuna.Study, best: optuna.trial.FrozenTrial) -> Path:
    out_dir = Path(args.logs_root) / args.study_name
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "study_name": args.study_name,
        "weighted": bool(args.use_correlation_weighting),
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "fixed_args": {
            "script": args.script,
            "optimizer": args.optimizer,
            "update_epochs": args.update_epochs,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "num_minibatches": args.num_minibatches,
            "total_timesteps": args.total_timesteps,
            "seeds": args.seeds,
            "metric": args.metric,
            "metric_window": args.metric_window,
        },
        "best_eval_command_template": (
            f"{sys.executable} {args.script} "
            "--env-id=<ENV_ID> --seed=<SEED> "
            f"--optimizer={args.optimizer} "
            f"--learning-rate={best.params.get('lr')} "
            f"--clip-coef={best.params.get('clip_coef')} "
            f"--ent-coef={best.params.get('ent_coef')} "
            f"--update-epochs={args.update_epochs} "
            f"{'--use-correlation-weighting' if args.use_correlation_weighting else '--no-use-correlation-weighting'}"
        ),
        "all_trials": [
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": dict(t.params),
                "user_attrs": dict(t.user_attrs),
                "intermediate_values": dict(t.intermediate_values),
            }
            for t in study.trials
        ],
    }

    out_path = out_dir / "best_result.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    txt_path = out_dir / "best_result.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"study_name: {args.study_name}\n")
        f.write(f"weighted: {bool(args.use_correlation_weighting)}\n")
        f.write(f"best_trial_number: {best.number}\n")
        f.write(f"best_value: {best.value}\n")
        f.write("best_params:\n")
        for k, v in best.params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nBest eval command template:\n")
        f.write(payload["best_eval_command_template"] + "\n")

    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--script", type=str, default="cleanrl/ppo_atari_envpool_corr_flag.py")
    p.add_argument("--metric", type=str, default="charts/episodic_return")
    p.add_argument("--metric-window", type=int, default=50)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--optimizer", type=str, default="RLMuon")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--study-name", type=str, default="corr_weighting_study")
    p.add_argument("--storage", type=str, default="sqlite:///corr_weighting_optuna.db")
    p.add_argument("--logs-root", type=str, default="tuner_logs")
    p.add_argument("--exp-name", type=str, default="corr_tune")
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=256)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--aux-learning-rate", type=float, default=None)

    p.add_argument("--use-correlation-weighting", dest="use_correlation_weighting", action="store_true")
    p.add_argument("--no-use-correlation-weighting", dest="use_correlation_weighting", action="store_false")
    p.set_defaults(use_correlation_weighting=True)

    p.add_argument("--track-runs", action="store_true", help="Pass --track through to training runs.")
    p.add_argument("--wandb-project-name", type=str, default="cleanRL")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tag", type=str, default=None)

    args = p.parse_args()

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpu_list:
        raise ValueError("No GPUs specified.")

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        sampler=optuna.samplers.TPESampler(seed=123),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = trial_params(trial)
        jobs = []
        for seed in range(args.seeds):
            for env_id in ATARI10:
                jobs.append((env_id, seed))

        raw_results = []
        norm_scores = []

        with ThreadPoolExecutor(max_workers=len(gpu_list)) as ex:
            futures = []
            for j, (env_id, seed) in enumerate(jobs):
                gpu = gpu_list[j % len(gpu_list)]
                futures.append(
                    ex.submit(
                        run_one,
                        args=args,
                        gpu=gpu,
                        env_id=env_id,
                        seed=seed,
                        params=params,
                        trial_number=trial.number,
                    )
                )

            for fut in as_completed(futures):
                env_id, seed, raw_score, norm_score, run_dir = fut.result()
                raw_results.append(
                    {
                        "env_id": env_id,
                        "seed": seed,
                        "raw_score": raw_score,
                        "normalized_score": norm_score,
                        "run_dir": run_dir,
                    }
                )
                norm_scores.append(norm_score)
                print(
                    f"[trial {trial.number}] env={env_id} seed={seed} "
                    f"raw={raw_score:.4f} norm={norm_score:.4f}"
                )

        # Aggregate per seed first, then average seeds, matching the usual Atari multi-seed idea.
        seed_scores = []
        for seed in range(args.seeds):
            vals = [r["normalized_score"] for r in raw_results if r["seed"] == seed]
            if vals:
                seed_scores.append(float(np.mean(vals)))
                trial.report(seed_scores[-1], step=seed)

        value = float(np.mean(seed_scores)) if seed_scores else float(np.mean(norm_scores))
        trial.set_user_attr("raw_results", raw_results)
        trial.set_user_attr("weighted", bool(args.use_correlation_weighting))

        if trial.should_prune():
            raise optuna.TrialPruned()

        return value

    study.optimize(objective, n_trials=args.trials)

    best_path = write_best_artifacts(args, study, study.best_trial)
    print(f"[done] study={args.study_name}")
    print(f"[done] weighted={bool(args.use_correlation_weighting)}")
    print(f"[done] best_value={study.best_trial.value}")
    print(f"[done] best_params={study.best_trial.params}")
    print(f"[done] best artifact: {best_path}")


if __name__ == "__main__":
    main()

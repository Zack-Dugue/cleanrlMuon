#!/usr/bin/env python3
"""Optimizer-specific hyperparameter tuning for continuous-control PQN.

Each Optuna trial is evaluated on every requested environment and seed.  Runs
are distributed across the listed GPUs, one subprocess at a time per GPU.  The
objective is the mean normalized final evaluation return, so HalfCheetah's
larger raw score cannot dominate Hopper or Walker2d.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

try:
    from .launch_utils import (
        DEFAULT_SCORE_RANGES,
        Device,
        normalized_score,
        parse_csv,
        parse_devices,
        run_training,
        safe_slug,
        write_json,
    )
except ImportError:
    from launch_utils import (  # type: ignore
        DEFAULT_SCORE_RANGES,
        Device,
        normalized_score,
        parse_csv,
        parse_devices,
        run_training,
        safe_slug,
        write_json,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = REPO_ROOT / "cleanrl" / "pqn_mujoco.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=("Adam", "Muon"), default="Muon")
    parser.add_argument(
        "--envs",
        default="HalfCheetah-v4,Hopper-v4,Walker2d-v4",
        help="Comma-separated tuning environments.",
    )
    parser.add_argument(
        "--gpus", default="0", help="Comma-separated GPU indices, or 'cpu'."
    )
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--seeds", type=int, default=2, help="Seeds per environment in each trial."
    )
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--study-name", default=None)
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna URL; defaults to SQLite in the study folder.",
    )
    parser.add_argument("--logs-root", default="mujoco/tuner_logs")
    parser.add_argument("--sampler-seed", type=int, default=2026)
    parser.add_argument("--search-space", choices=("core", "full"), default="core")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--score-ranges-json", default=None)
    parser.add_argument("--cpus-per-run", type=int, default=2)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project-name", default="cleanRL-mujoco-pqn-tuning")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def fixed_params(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "total-timesteps": args.total_timesteps,
        "num-envs": args.num_envs,
        "num-steps": args.num_steps,
        "num-minibatches": args.num_minibatches,
        "eval-episodes": args.eval_episodes,
        "optimizer": args.optimizer,
        "norm-type": "layernorm",
        "obs-norm": True,
        "anneal-lr": False,
        "muon-actor": False,
    }


def suggest_params(trial, optimizer: str, search_space: str) -> Dict[str, object]:
    if optimizer == "Muon":
        critic_lr = trial.suggest_float("critic_learning_rate", 1e-4, 3e-2, log=True)
    else:
        critic_lr = trial.suggest_float("critic_learning_rate", 3e-5, 3e-3, log=True)
    actor_lr = trial.suggest_float("actor_learning_rate", 3e-5, 3e-3, log=True)
    distance_from_one = trial.suggest_float("distance_from_one", 0.01, 0.50, log=True)
    start_noise = trial.suggest_float("start_noise", 0.10, 0.50)
    end_noise_fraction = trial.suggest_float("end_noise_fraction", 0.10, 0.60)

    result: Dict[str, object] = {
        "critic-learning-rate": critic_lr,
        "actor-learning-rate": actor_lr,
        "q-lambda": 1.0 - distance_from_one,
        "start-noise": start_noise,
        "end-noise": start_noise * end_noise_fraction,
        "exploration-fraction": 0.60,
        "update-epochs": 2,
        "actor-update-frequency": 1,
        "hidden-dim": 256,
    }
    if search_space == "full":
        result.update(
            {
                "exploration-fraction": trial.suggest_float(
                    "exploration_fraction", 0.25, 1.0
                ),
                "update-epochs": trial.suggest_categorical("update_epochs", [1, 2, 4]),
                "actor-update-frequency": trial.suggest_categorical(
                    "actor_update_frequency", [1, 2, 4]
                ),
                "hidden-dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
            }
        )
    return result


def resolve_saved_params(
    raw_params: Mapping[str, object], optimizer: str, search_space: str
) -> Dict[str, object]:
    start_noise = float(raw_params["start_noise"])
    result: Dict[str, object] = {
        "critic-learning-rate": float(raw_params["critic_learning_rate"]),
        "actor-learning-rate": float(raw_params["actor_learning_rate"]),
        "q-lambda": 1.0 - float(raw_params["distance_from_one"]),
        "start-noise": start_noise,
        "end-noise": start_noise * float(raw_params["end_noise_fraction"]),
        "exploration-fraction": 0.60,
        "update-epochs": 2,
        "actor-update-frequency": 1,
        "hidden-dim": 256,
    }
    if search_space == "full":
        result.update(
            {
                "exploration-fraction": float(raw_params["exploration_fraction"]),
                "update-epochs": int(raw_params["update_epochs"]),
                "actor-update-frequency": int(raw_params["actor_update_frequency"]),
                "hidden-dim": int(raw_params["hidden_dim"]),
            }
        )
    return result


def load_score_ranges(path: Optional[str]) -> Dict[str, List[float]]:
    ranges = {key: list(value) for key, value in DEFAULT_SCORE_RANGES.items()}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            overrides = json.load(handle)
        ranges.update(
            {
                key: [float(value[0]), float(value[1])]
                for key, value in overrides.items()
            }
        )
    return ranges


def distribute(
    items: Sequence[Dict[str, object]], devices: Sequence[Device]
) -> Dict[Device, List[Dict[str, object]]]:
    queues: Dict[Device, List[Dict[str, object]]] = {device: [] for device in devices}
    for index, item in enumerate(items):
        queues[devices[index % len(devices)]].append(item)
    return queues


def main() -> None:
    args = parse_args()
    environments = parse_csv(args.envs)
    devices = parse_devices(args.gpus)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    if not environments:
        raise ValueError("At least one environment is required")
    score_ranges = load_score_ranges(args.score_ranges_json)
    for environment in environments:
        normalized_score(environment, 0.0, score_ranges)

    study_name = (
        args.study_name or f"pqn_mujoco_{args.optimizer.lower()}_{args.search_space}"
    )
    study_dir = Path(args.logs_root) / safe_slug(study_name)
    study_dir.mkdir(parents=True, exist_ok=True)
    common_params = fixed_params(args)

    representative = {
        "critic-learning-rate": 3e-3 if args.optimizer == "Muon" else 3e-4,
        "actor-learning-rate": 3e-4,
        "q-lambda": 0.95,
        "start-noise": 0.30,
        "end-noise": 0.06,
        "exploration-fraction": 0.60,
        "update-epochs": 2,
        "actor-update-frequency": 1,
        "hidden-dim": 256,
    }
    if args.dry_run:
        dry_tasks = [
            {"env_id": env_id, "seed": seed}
            for env_id in environments
            for seed in seeds
        ]
        for index, task in enumerate(dry_tasks):
            env_id = str(task["env_id"])
            seed = int(task["seed"])
            result = run_training(
                script=TRAINING_SCRIPT,
                params={**common_params, **representative},
                env_id=env_id,
                seed=seed,
                device=devices[index % len(devices)],
                output_dir=study_dir / "dry_run" / f"{safe_slug(env_id)}__seed{seed}",
                log_path=study_dir / "dry_run" / f"{safe_slug(env_id)}__seed{seed}.log",
                track=args.track,
                wandb_project_name=args.wandb_project_name,
                wandb_entity=args.wandb_entity,
                wandb_group=study_name,
                cpus_per_run=args.cpus_per_run,
                dry_run=True,
            )
            print(result["command"])
        return

    try:
        import optuna
    except ImportError as error:
        raise SystemExit(
            "Install the tuning dependencies with: pip install -e '.[mujoco,optuna]'"
        ) from error

    storage = args.storage or f"sqlite:///{(study_dir / 'study.db').resolve()}"
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial) -> float:
        sampled_params = suggest_params(trial, args.optimizer, args.search_space)
        trial_params = {**common_params, **sampled_params}
        tasks = [
            {"env_id": env_id, "seed": seed}
            for env_id in environments
            for seed in seeds
        ]
        queues = distribute(tasks, devices)
        trial_dir = study_dir / f"trial_{trial.number:05d}"

        def run_queue(
            device: Device, queue: Sequence[Dict[str, object]]
        ) -> List[Dict[str, object]]:
            results: List[Dict[str, object]] = []
            for task in queue:
                env_id = str(task["env_id"])
                seed = int(task["seed"])
                slug = f"{safe_slug(env_id)}__seed{seed}"
                results.append(
                    run_training(
                        script=TRAINING_SCRIPT,
                        params=trial_params,
                        env_id=env_id,
                        seed=seed,
                        device=device,
                        output_dir=trial_dir / "runs" / slug,
                        log_path=trial_dir / "logs" / f"{slug}.log",
                        track=args.track,
                        wandb_project_name=args.wandb_project_name,
                        wandb_entity=args.wandb_entity,
                        wandb_group=study_name,
                        cpus_per_run=args.cpus_per_run,
                        dry_run=False,
                    )
                )
            return results

        all_results: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(run_queue, device, queue)
                for device, queue in queues.items()
            ]
            for future in as_completed(futures):
                all_results.extend(future.result())

        failures = [result for result in all_results if result.get("status") != "ok"]
        normalized_results = [
            normalized_score(
                str(result["env_id"]), float(result["eval_mean_return"]), score_ranges
            )
            for result in all_results
            if result.get("status") == "ok"
        ]
        payload = {
            "trial_number": trial.number,
            "optimizer": args.optimizer,
            "sampled_params": sampled_params,
            "raw_optuna_params": dict(trial.params),
            "normalized_scores": normalized_results,
            "objective": float(sum(normalized_results) / len(normalized_results))
            if normalized_results
            else -math.inf,
            "failures": len(failures),
            "runs": all_results,
        }
        write_json(trial_dir / "trial_summary.json", payload)
        trial.set_user_attr("failed_runs", len(failures))
        if failures:
            print(
                f"trial {trial.number}: {len(failures)} failed run(s); see {trial_dir}"
            )
            return -1e9
        value = float(sum(normalized_results) / len(normalized_results))
        print(
            f"trial {trial.number}: normalized return={value:.6f} params={sampled_params}"
        )
        return value

    study.optimize(objective, n_trials=args.trials)
    best = study.best_trial
    resolved_best = resolve_saved_params(best.params, args.optimizer, args.search_space)
    best_payload = {
        "schema_version": 1,
        "algorithm": "actor_critic_pqn",
        "study_name": study_name,
        "optimizer": args.optimizer,
        "metric": "mean_normalized_eval_return",
        "best_value": float(best.value),
        "best_trial_number": int(best.number),
        "best_params": resolved_best,
        "raw_optuna_params": dict(best.params),
        "fixed_params": common_params,
        "environments": environments,
        "score_ranges": {
            environment: score_ranges[environment] for environment in environments
        },
        "seeds_per_trial": args.seeds,
        "search_space": args.search_space,
        "training_script": str(TRAINING_SCRIPT.relative_to(REPO_ROOT)),
    }
    best_path = study_dir / "best_hyperparams.json"
    write_json(best_path, best_payload)
    with (study_dir / "best_hyperparams.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"study: {study_name}\noptimizer: {args.optimizer}\n")
        handle.write(
            f"best trial: {best.number}\nbest normalized return: {best.value:.8f}\n"
        )
        handle.write("best parameters:\n")
        for key, value in resolved_best.items():
            handle.write(f"  {key}: {value}\n")
    print(f"best normalized return: {best.value:.8f}")
    print(f"best parameters: {resolved_best}")
    print(f"saved: {best_path.resolve()}")


if __name__ == "__main__":
    main()

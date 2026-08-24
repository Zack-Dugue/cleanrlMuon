#!/usr/bin/env python3
"""Optimizer-specific Optuna tuning for discrete DeepSea PQN."""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

try:
    from .launch_utils import (
        Device,
        normalized_return,
        parse_devices,
        parse_sizes,
        run_training,
        safe_slug,
        write_json,
    )
except ImportError:
    from launch_utils import (  # type: ignore
        Device,
        normalized_return,
        parse_devices,
        parse_sizes,
        run_training,
        safe_slug,
        write_json,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = REPO_ROOT / "cleanrl" / "pqn_deepsea.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=("Adam", "Muon"), default="Muon")
    parser.add_argument("--sizes", default="10,20,30")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--logs-root", default="logs/deepsea/tuning")
    parser.add_argument("--sampler-seed", type=int, default=2026)
    parser.add_argument("--search-space", choices=("core", "full"), default="core")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=2048)
    parser.add_argument("--cpus-per-run", type=int, default=2)
    track_group = parser.add_mutually_exclusive_group()
    track_group.add_argument("--track", dest="track", action="store_true")
    track_group.add_argument("--no-track", dest="track", action="store_false")
    parser.set_defaults(track=True)
    parser.add_argument(
        "--wandb-project-name", default="cleanRL-deepsea-pqn-tuning"
    )
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
        "anneal-lr": True,
        "norm-type": "layernorm",
        "hidden-layers": 2,
        "use-muon-input": True,
        "use-muon-output": False,
    }


def suggest_params(trial, optimizer: str, search_space: str) -> Dict[str, object]:
    if optimizer == "Muon":
        learning_rate = trial.suggest_float("learning_rate", 3.0e-4, 3.0e-2, log=True)
    else:
        learning_rate = trial.suggest_float("learning_rate", 3.0e-5, 3.0e-3, log=True)
    distance_from_one = trial.suggest_float("distance_from_one", 1.0e-3, 1.0, log=True)
    result: Dict[str, object] = {
        "learning-rate": learning_rate,
        "q-lambda": 1.0 - distance_from_one,
        "exploration-fraction": trial.suggest_float(
            "exploration_fraction", 0.03, 0.50
        ),
        "start-e": 1.0,
        "end-e": 0.01,
        "update-epochs": 4,
        "hidden-dim": 256,
    }
    if search_space == "full":
        result.update(
            {
                "end-e": trial.suggest_float("end_e", 0.0, 0.20),
                "update-epochs": trial.suggest_categorical("update_epochs", [1, 2, 4]),
                "hidden-dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
            }
        )
    return result


def resolve_saved_params(
    raw_params: Mapping[str, object], optimizer: str, search_space: str
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "learning-rate": float(raw_params["learning_rate"]),
        "q-lambda": 1.0 - float(raw_params["distance_from_one"]),
        "exploration-fraction": float(raw_params["exploration_fraction"]),
        "start-e": 1.0,
        "end-e": 0.01,
        "update-epochs": 4,
        "hidden-dim": 256,
    }
    if search_space == "full":
        result.update(
            {
                "end-e": float(raw_params["end_e"]),
                "update-epochs": int(raw_params["update_epochs"]),
                "hidden-dim": int(raw_params["hidden_dim"]),
            }
        )
    return result


def distribute(
    items: Sequence[Dict[str, object]], devices: Sequence[Device]
) -> Dict[Device, List[Dict[str, object]]]:
    queues: Dict[Device, List[Dict[str, object]]] = {device: [] for device in devices}
    for index, item in enumerate(items):
        queues[devices[index % len(devices)]].append(item)
    return queues


def main() -> None:
    args = parse_args()
    sizes = parse_sizes(args.sizes)
    devices = parse_devices(args.gpus)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    study_name = args.study_name or f"deepsea_pqn_{args.optimizer.lower()}_{args.search_space}"
    study_dir = Path(args.logs_root) / safe_slug(study_name)
    study_dir.mkdir(parents=True, exist_ok=True)
    common_params = fixed_params(args)

    representative = {
        "learning-rate": 3.0e-3 if args.optimizer == "Muon" else 3.0e-4,
        "q-lambda": 0.65,
        "exploration-fraction": 0.10,
        "start-e": 1.0,
        "end-e": 0.01,
        "update-epochs": 4,
        "hidden-dim": 256,
    }
    if args.dry_run:
        for index, (size, seed) in enumerate(
            (size, seed) for size in sizes for seed in seeds
        ):
            result = run_training(
                script=TRAINING_SCRIPT,
                params={**common_params, **representative},
                deepsea_size=size,
                seed=seed,
                device=devices[index % len(devices)],
                output_dir=study_dir / "dry_run" / f"size_{size}__seed_{seed}",
                log_path=study_dir / "dry_run" / f"size_{size}__seed_{seed}.log",
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
        raise SystemExit("Install the tuner with: pip install -e '.[optuna]'") from error

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
        tuned_params = suggest_params(trial, args.optimizer, args.search_space)
        tasks = [
            {"deepsea_size": size, "seed": seed}
            for size in sizes
            for seed in seeds
        ]
        queues = distribute(tasks, devices)

        def run_queue(
            device: Device, queue: Sequence[Dict[str, object]]
        ) -> List[Dict[str, object]]:
            results: List[Dict[str, object]] = []
            for task in queue:
                size = int(task["deepsea_size"])
                seed = int(task["seed"])
                run_dir = (
                    study_dir
                    / "trials"
                    / f"trial_{trial.number:04d}"
                    / f"size_{size}__seed_{seed}"
                )
                results.append(
                    run_training(
                        script=TRAINING_SCRIPT,
                        params={**common_params, **tuned_params},
                        deepsea_size=size,
                        seed=seed,
                        device=device,
                        output_dir=run_dir,
                        log_path=run_dir.with_suffix(".log"),
                        track=args.track,
                        wandb_project_name=args.wandb_project_name,
                        wandb_entity=args.wandb_entity,
                        wandb_group=f"{study_name}-trial-{trial.number}",
                        cpus_per_run=args.cpus_per_run,
                        dry_run=False,
                    )
                )
            return results

        results: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(run_queue, device, queue)
                for device, queue in queues.items()
            ]
            for future in as_completed(futures):
                results.extend(future.result())
        failures = [result for result in results if result.get("status") != "ok"]
        trial_dir = study_dir / "trials" / f"trial_{trial.number:04d}"
        write_json(
            trial_dir / "trial_results.json",
            {
                "trial_number": trial.number,
                "optimizer": args.optimizer,
                "params": tuned_params,
                "results": results,
            },
        )
        if failures:
            messages = "\n\n".join(str(result.get("error")) for result in failures)
            raise RuntimeError(f"{len(failures)} DeepSea run(s) failed:\n{messages}")
        scores = [
            normalized_return(
                float(result["eval_greedy_return"]), int(result["deepsea_size"])
            )
            for result in results
        ]
        value = float(statistics.mean(scores))
        trial.set_user_attr("size_seed_scores", scores)
        trial.set_user_attr(
            "mean_greedy_success",
            float(statistics.mean(float(result["eval_greedy_success"]) for result in results)),
        )
        return value

    study.optimize(objective, n_trials=args.trials)
    best_params = resolve_saved_params(
        study.best_trial.params, args.optimizer, args.search_space
    )
    payload = {
        "algorithm": "discrete_pqn",
        "environment": "DeepSea",
        "optimizer": args.optimizer,
        "study_name": study.study_name,
        "search_space": args.search_space,
        "sizes": sizes,
        "seeds_per_trial": args.seeds,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": best_params,
        "optuna_params": dict(study.best_trial.params),
        "fixed_params": common_params,
        "objective": "mean normalized greedy return across matched sizes and seeds",
    }
    output_path = study_dir / "best_hyperparams.json"
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

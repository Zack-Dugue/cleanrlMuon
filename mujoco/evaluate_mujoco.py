#!/usr/bin/env python3
"""Evaluate one or more tuned MuJoCo PQN configurations.

The input is the ``best_hyperparams.json`` written by ``tune_mujoco.py``.
Optimizer-specific configurations are intentionally kept separate: this tests
Adam and Muon at their own tuned settings rather than silently giving one
optimizer the other's learning rate.  The launcher resumes completed runs and
writes raw, aggregate, and seed-paired comparison CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

try:
    from .launch_utils import (
        DEFAULT_SCORE_RANGES,
        Device,
        normalized_score,
        parse_csv,
        parse_devices,
        read_json,
        run_training,
        safe_slug,
    )
except ImportError:
    from launch_utils import (  # type: ignore
        DEFAULT_SCORE_RANGES,
        Device,
        normalized_score,
        parse_csv,
        parse_devices,
        read_json,
        run_training,
        safe_slug,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = REPO_ROOT / "cleanrl" / "pqn_mujoco.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="One or more best_hyperparams.json files (normally one per optimizer).",
    )
    parser.add_argument(
        "--envs",
        default=None,
        help="Optional comma-separated evaluation environments; defaults to the union in the configs.",
    )
    parser.add_argument(
        "--gpus", default="0", help="Comma-separated GPU indices, or 'cpu'."
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--output-root", default="mujoco/evaluation")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--greedy-eval-steps",
        type=int,
        default=50_000,
        help="Total noise-free evaluation transitions after training.",
    )
    parser.add_argument(
        "--greedy-eval-num-envs",
        type=int,
        default=8,
        help="Parallel environments used for fixed-budget greedy evaluation.",
    )
    parser.add_argument("--cpus-per-run", type=int, default=2)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project-name", default="cleanRL-mujoco-pqn-evaluation")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="mujoco-pqn-final-evaluation")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun tasks with an existing summary.json.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def load_configs(paths: Sequence[str]) -> List[Dict[str, object]]:
    configurations: List[Dict[str, object]] = []
    seen_optimizers = set()
    for path_string in paths:
        path = Path(path_string).resolve()
        payload = read_json(path)
        if payload.get("algorithm") != "actor_critic_pqn":
            raise ValueError(f"{path} is not a mujoco_pqn tuning result")
        optimizer = str(payload["optimizer"])
        if optimizer in seen_optimizers:
            raise ValueError(
                f"Multiple configs use optimizer {optimizer!r}; optimizer labels must be unique"
            )
        seen_optimizers.add(optimizer)
        params = dict(payload.get("fixed_params", {}))
        params.update(dict(payload.get("best_params", {})))
        params["optimizer"] = optimizer
        configurations.append(
            {
                "path": str(path),
                "optimizer": optimizer,
                "params": params,
                "environments": list(payload.get("environments", [])),
                "score_ranges": dict(payload.get("score_ranges", {})),
                "best_value": payload.get("best_value"),
                "best_trial_number": payload.get("best_trial_number"),
            }
        )
    return configurations


def distribute(
    items: Sequence[Dict[str, object]], devices: Sequence[Device]
) -> Dict[Device, List[Dict[str, object]]]:
    queues: Dict[Device, List[Dict[str, object]]] = {device: [] for device in devices}
    for index, item in enumerate(items):
        queues[devices[index % len(devices)]].append(item)
    return queues


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return (
        float(statistics.mean(values)),
        float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    )


def performance_return(result: Mapping[str, object], use_greedy_eval: bool) -> float:
    metric = "greedy_eval_mean_return" if use_greedy_eval else "eval_mean_return"
    if result.get(metric) is None:
        raise ValueError(f"Completed run is missing required metric {metric!r}")
    return float(result[metric])


def main() -> None:
    args = parse_args()
    configurations = load_configs(args.configs)
    devices = parse_devices(args.gpus)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    if args.envs:
        environments = parse_csv(args.envs)
    else:
        environments = sorted(
            {
                environment
                for configuration in configurations
                for environment in configuration["environments"]
            }
        )
    if not environments:
        raise ValueError(
            "No environments were provided or found in the tuning configurations"
        )

    score_ranges = {key: list(value) for key, value in DEFAULT_SCORE_RANGES.items()}
    for configuration in configurations:
        score_ranges.update(
            {
                key: [float(value[0]), float(value[1])]
                for key, value in dict(configuration["score_ranges"]).items()
            }
        )
    for environment in environments:
        normalized_score(environment, 0.0, score_ranges)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks: List[Dict[str, object]] = []
    for configuration in configurations:
        for environment in environments:
            for seed in seeds:
                tasks.append(
                    {
                        "configuration": configuration,
                        "env_id": environment,
                        "seed": seed,
                    }
                )
    queues = distribute(tasks, devices)

    def run_queue(
        device: Device, queue: Sequence[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for task in queue:
            configuration = dict(task["configuration"])
            optimizer = str(configuration["optimizer"])
            env_id = str(task["env_id"])
            seed = int(task["seed"])
            params = dict(configuration["params"])
            params["eval-episodes"] = args.eval_episodes
            params["greedy-eval-steps"] = args.greedy_eval_steps
            params["greedy-eval-num-envs"] = args.greedy_eval_num_envs
            if args.total_timesteps is not None:
                params["total-timesteps"] = args.total_timesteps
            run_dir = (
                output_root
                / "runs"
                / safe_slug(optimizer)
                / safe_slug(env_id)
                / f"seed_{seed}"
            )
            summary_path = run_dir / "summary.json"
            resumable_summary = None
            if summary_path.exists() and not args.overwrite and not args.dry_run:
                candidate_summary = read_json(summary_path)
                completed_greedy_steps = int(
                    candidate_summary.get("greedy_eval_steps_actual", 0)
                )
                if (
                    args.greedy_eval_steps == 0
                    or completed_greedy_steps >= args.greedy_eval_steps
                ):
                    resumable_summary = candidate_summary
            if resumable_summary is not None:
                result = resumable_summary
                result.update(
                    {
                        "status": "ok",
                        "resumed": True,
                        "config_path": configuration["path"],
                        "output_dir": str(run_dir),
                    }
                )
            else:
                result = run_training(
                    script=TRAINING_SCRIPT,
                    params=params,
                    env_id=env_id,
                    seed=seed,
                    device=device,
                    output_dir=run_dir,
                    log_path=output_root
                    / "logs"
                    / safe_slug(optimizer)
                    / f"{safe_slug(env_id)}__seed{seed}.log",
                    track=args.track,
                    wandb_project_name=args.wandb_project_name,
                    wandb_entity=args.wandb_entity,
                    wandb_group=args.wandb_group,
                    cpus_per_run=args.cpus_per_run,
                    dry_run=args.dry_run,
                )
                result["resumed"] = False
                result["config_path"] = configuration["path"]
            results.append(result)
        return results

    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(run_queue, device, queue)
            for device, queue in queues.items()
        ]
        for future in as_completed(futures):
            results.extend(future.result())

    if args.dry_run:
        for result in results:
            print(result["command"])
        return

    successful = [result for result in results if result.get("status") == "ok"]
    failures = [result for result in results if result.get("status") != "ok"]
    raw_rows: List[Dict[str, object]] = []
    for result in results:
        row = {
            "status": result.get("status"),
            "optimizer": result.get("optimizer"),
            "env_id": result.get("env_id"),
            "seed": result.get("seed"),
            "eval_mean_return": result.get("eval_mean_return"),
            "eval_median_return": result.get("eval_median_return"),
            "eval_std_return": result.get("eval_std_return"),
            "greedy_eval_steps_actual": result.get("greedy_eval_steps_actual"),
            "greedy_eval_episodes": result.get("greedy_eval_episodes"),
            "greedy_eval_mean_return": result.get("greedy_eval_mean_return"),
            "greedy_eval_median_return": result.get("greedy_eval_median_return"),
            "greedy_eval_std_return": result.get("greedy_eval_std_return"),
            "greedy_eval_seconds": result.get("greedy_eval_seconds"),
            "training_return_mean_last_20": result.get("training_return_mean_last_20"),
            "training_seconds": result.get("training_seconds"),
            "sps": result.get("sps"),
            "global_step": result.get("global_step"),
            "resumed": result.get("resumed"),
            "config_path": result.get("config_path"),
            "output_dir": result.get("output_dir"),
            "log_path": result.get("log_path"),
            "error": result.get("error"),
        }
        if result.get("status") == "ok":
            primary_return = performance_return(
                result, use_greedy_eval=args.greedy_eval_steps > 0
            )
            row["performance_metric"] = (
                "greedy_eval_mean_return"
                if args.greedy_eval_steps > 0
                else "eval_mean_return"
            )
            row["performance_return"] = primary_return
            row["normalized_eval_return"] = normalized_score(
                str(result["env_id"]), primary_return, score_ranges
            )
        raw_rows.append(row)
    raw_fields = [
        "status",
        "optimizer",
        "env_id",
        "seed",
        "eval_mean_return",
        "normalized_eval_return",
        "eval_median_return",
        "eval_std_return",
        "greedy_eval_steps_actual",
        "greedy_eval_episodes",
        "greedy_eval_mean_return",
        "greedy_eval_median_return",
        "greedy_eval_std_return",
        "greedy_eval_seconds",
        "performance_metric",
        "performance_return",
        "training_return_mean_last_20",
        "training_seconds",
        "sps",
        "global_step",
        "resumed",
        "config_path",
        "output_dir",
        "log_path",
        "error",
    ]
    write_csv(output_root / "run_results.csv", raw_rows, raw_fields)

    environment_rows: List[Dict[str, object]] = []
    for optimizer in sorted({str(result["optimizer"]) for result in successful}):
        for env_id in environments:
            matching = [
                result
                for result in successful
                if result["optimizer"] == optimizer and result["env_id"] == env_id
            ]
            returns = [
                performance_return(result, use_greedy_eval=args.greedy_eval_steps > 0)
                for result in matching
            ]
            normalized = [
                normalized_score(env_id, value, score_ranges) for value in returns
            ]
            return_mean, return_std = mean_std(returns)
            normalized_mean, normalized_std = mean_std(normalized)
            environment_rows.append(
                {
                    "optimizer": optimizer,
                    "env_id": env_id,
                    "performance_metric": (
                        "greedy_eval_mean_return"
                        if args.greedy_eval_steps > 0
                        else "eval_mean_return"
                    ),
                    "successful_seeds": len(matching),
                    "mean_return": return_mean,
                    "std_return": return_std,
                    "median_return": float(statistics.median(returns))
                    if returns
                    else float("nan"),
                    "mean_normalized_return": normalized_mean,
                    "std_normalized_return": normalized_std,
                }
            )
    write_csv(
        output_root / "environment_summary.csv",
        environment_rows,
        [
            "optimizer",
            "env_id",
            "performance_metric",
            "successful_seeds",
            "mean_return",
            "std_return",
            "median_return",
            "mean_normalized_return",
            "std_normalized_return",
        ],
    )

    overall_rows: List[Dict[str, object]] = []
    for optimizer in sorted({str(result["optimizer"]) for result in successful}):
        matching = [result for result in successful if result["optimizer"] == optimizer]
        normalized = [
            normalized_score(
                str(result["env_id"]),
                performance_return(result, use_greedy_eval=args.greedy_eval_steps > 0),
                score_ranges,
            )
            for result in matching
        ]
        normalized_mean, normalized_std = mean_std(normalized)
        overall_rows.append(
            {
                "optimizer": optimizer,
                "performance_metric": (
                    "greedy_eval_mean_return"
                    if args.greedy_eval_steps > 0
                    else "eval_mean_return"
                ),
                "successful_runs": len(matching),
                "failed_runs": sum(
                    1
                    for result in failures
                    if str(result.get("optimizer")) == optimizer
                ),
                "mean_normalized_return": normalized_mean,
                "std_normalized_return": normalized_std,
                "mean_training_seconds": mean_std(
                    [float(result["training_seconds"]) for result in matching]
                )[0],
                "mean_sps": mean_std([float(result["sps"]) for result in matching])[0],
            }
        )
    write_csv(
        output_root / "overall_summary.csv",
        overall_rows,
        [
            "optimizer",
            "performance_metric",
            "successful_runs",
            "failed_runs",
            "mean_normalized_return",
            "std_normalized_return",
            "mean_training_seconds",
            "mean_sps",
        ],
    )

    optimizers = [str(configuration["optimizer"]) for configuration in configurations]
    baseline = "Adam" if "Adam" in optimizers else optimizers[0]
    lookup = {
        (str(result["optimizer"]), str(result["env_id"]), int(result["seed"])): result
        for result in successful
    }
    paired_rows: List[Dict[str, object]] = []
    for contender in optimizers:
        if contender == baseline:
            continue
        for env_id in environments:
            for seed in seeds:
                baseline_result = lookup.get((baseline, env_id, seed))
                contender_result = lookup.get((contender, env_id, seed))
                if baseline_result is None or contender_result is None:
                    continue
                baseline_return = performance_return(
                    baseline_result, use_greedy_eval=args.greedy_eval_steps > 0
                )
                contender_return = performance_return(
                    contender_result, use_greedy_eval=args.greedy_eval_steps > 0
                )
                paired_rows.append(
                    {
                        "baseline": baseline,
                        "contender": contender,
                        "env_id": env_id,
                        "seed": seed,
                        "performance_metric": (
                            "greedy_eval_mean_return"
                            if args.greedy_eval_steps > 0
                            else "eval_mean_return"
                        ),
                        "baseline_return": baseline_return,
                        "contender_return": contender_return,
                        "raw_delta": contender_return - baseline_return,
                        "normalized_delta": normalized_score(
                            env_id, contender_return, score_ranges
                        )
                        - normalized_score(env_id, baseline_return, score_ranges),
                    }
                )
    write_csv(
        output_root / "paired_deltas.csv",
        paired_rows,
        [
            "baseline",
            "contender",
            "env_id",
            "seed",
            "performance_metric",
            "baseline_return",
            "contender_return",
            "raw_delta",
            "normalized_delta",
        ],
    )

    print(f"completed {len(successful)}/{len(results)} runs; failures={len(failures)}")
    print(f"results: {(output_root / 'overall_summary.csv').resolve()}")
    if failures:
        print(f"failure details: {(output_root / 'run_results.csv').resolve()}")


if __name__ == "__main__":
    main()

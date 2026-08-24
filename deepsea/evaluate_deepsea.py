#!/usr/bin/env python3
"""Evaluate tuned Adam and Muon DeepSea PQN configurations on matched seeds."""

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
        Device,
        normalized_return,
        parse_devices,
        parse_sizes,
        read_json,
        run_training,
        safe_slug,
    )
except ImportError:
    from launch_utils import (  # type: ignore
        Device,
        normalized_return,
        parse_devices,
        parse_sizes,
        read_json,
        run_training,
        safe_slug,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = REPO_ROOT / "cleanrl" / "pqn_deepsea.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--sizes", default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--output-root", default="logs/deepsea/evaluation")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=4096)
    parser.add_argument("--cpus-per-run", type=int, default=2)
    track_group = parser.add_mutually_exclusive_group()
    track_group.add_argument("--track", dest="track", action="store_true")
    track_group.add_argument("--no-track", dest="track", action="store_false")
    parser.set_defaults(track=True)
    parser.add_argument(
        "--wandb-project-name", default="cleanRL-deepsea-pqn-evaluation"
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="deepsea-pqn-final-evaluation")
    parser.add_argument("--overwrite", action="store_true")
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
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def load_configs(paths: Sequence[str]) -> List[Dict[str, object]]:
    configurations: List[Dict[str, object]] = []
    seen = set()
    for path_string in paths:
        path = Path(path_string).resolve()
        payload = read_json(path)
        if payload.get("algorithm") != "discrete_pqn" or payload.get("environment") != "DeepSea":
            raise ValueError(f"{path} is not a DeepSea discrete-PQN tuning result")
        optimizer = str(payload["optimizer"])
        if optimizer in seen:
            raise ValueError(f"More than one config uses optimizer {optimizer!r}")
        seen.add(optimizer)
        params = dict(payload.get("fixed_params", {}))
        params.update(dict(payload.get("best_params", {})))
        params["optimizer"] = optimizer
        configurations.append(
            {
                "path": str(path),
                "optimizer": optimizer,
                "params": params,
                "sizes": [int(size) for size in payload.get("sizes", [])],
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


def summary_matches_request(
    summary: Mapping[str, object], params: Mapping[str, object], size: int, seed: int
) -> bool:
    if summary.get("status") != "ok":
        return False
    if int(summary.get("deepsea_size", -1)) != size or int(summary.get("seed", -1)) != seed:
        return False
    saved_config = summary.get("config")
    if not isinstance(saved_config, dict):
        return False
    for key, requested in params.items():
        normalized_key = str(key).replace("-", "_")
        if normalized_key not in saved_config or saved_config[normalized_key] != requested:
            return False
    return True


def main() -> None:
    args = parse_args()
    configurations = load_configs(args.configs)
    devices = parse_devices(args.gpus)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    if args.sizes:
        sizes = parse_sizes(args.sizes)
    else:
        sizes = sorted(
            {
                int(size)
                for configuration in configurations
                for size in configuration["sizes"]
            }
        )
    if not sizes:
        raise ValueError("No DeepSea sizes were supplied or found in the configs")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        {"configuration": configuration, "deepsea_size": size, "seed": seed}
        for configuration in configurations
        for size in sizes
        for seed in seeds
    ]
    queues = distribute(tasks, devices)

    def run_queue(
        device: Device, queue: Sequence[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for task in queue:
            configuration = dict(task["configuration"])
            optimizer = str(configuration["optimizer"])
            size = int(task["deepsea_size"])
            seed = int(task["seed"])
            params = dict(configuration["params"])
            params["eval-episodes"] = args.eval_episodes
            if args.total_timesteps is not None:
                params["total-timesteps"] = args.total_timesteps
            run_dir = (
                output_root
                / "runs"
                / safe_slug(optimizer)
                / f"size_{size}"
                / f"seed_{seed}"
            )
            summary_path = run_dir / "summary.json"
            if (
                summary_path.exists()
                and not args.overwrite
                and not args.dry_run
                and summary_matches_request(read_json(summary_path), params, size, seed)
            ):
                result = read_json(summary_path)
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
                    deepsea_size=size,
                    seed=seed,
                    device=device,
                    output_dir=run_dir,
                    log_path=output_root
                    / "launcher_logs"
                    / f"{safe_slug(optimizer)}__size_{size}__seed_{seed}.log",
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
            executor.submit(run_queue, device, queue) for device, queue in queues.items()
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(
        key=lambda item: (
            str(item.get("optimizer")),
            int(item.get("deepsea_size", -1)),
            int(item.get("seed", -1)),
        )
    )

    run_rows: List[Dict[str, object]] = []
    for result in results:
        row = dict(result)
        if result.get("status") == "ok":
            row["normalized_return"] = normalized_return(
                float(result["eval_greedy_return"]),
                int(result["deepsea_size"]),
            )
        run_rows.append(row)
    run_fields = [
        "optimizer",
        "deepsea_size",
        "seed",
        "status",
        "resumed",
        "eval_greedy_return",
        "normalized_return",
        "eval_greedy_success",
        "eval_optimal_action_accuracy",
        "sps",
        "elapsed_seconds",
        "output_dir",
        "log_path",
        "config_path",
        "error",
    ]
    write_csv(output_root / "run_results.csv", run_rows, run_fields)

    successful = [row for row in run_rows if row.get("status") == "ok"]
    size_rows: List[Dict[str, object]] = []
    for configuration in configurations:
        optimizer = str(configuration["optimizer"])
        for size in sizes:
            group = [
                row
                for row in successful
                if row.get("optimizer") == optimizer
                and int(row.get("deepsea_size", -1)) == size
            ]
            returns = [float(row["eval_greedy_return"]) for row in group]
            normalized = [float(row["normalized_return"]) for row in group]
            successes = [float(row["eval_greedy_success"]) for row in group]
            accuracies = [float(row["eval_optimal_action_accuracy"]) for row in group]
            mean_return, std_return = mean_std(returns)
            mean_normalized, std_normalized = mean_std(normalized)
            mean_success, std_success = mean_std(successes)
            mean_accuracy, std_accuracy = mean_std(accuracies)
            size_rows.append(
                {
                    "optimizer": optimizer,
                    "deepsea_size": size,
                    "completed_seeds": len(group),
                    "mean_greedy_return": mean_return,
                    "std_greedy_return": std_return,
                    "mean_normalized_return": mean_normalized,
                    "std_normalized_return": std_normalized,
                    "mean_greedy_success": mean_success,
                    "std_greedy_success": std_success,
                    "mean_action_accuracy": mean_accuracy,
                    "std_action_accuracy": std_accuracy,
                }
            )
    size_fields = list(size_rows[0].keys()) if size_rows else []
    write_csv(output_root / "size_summary.csv", size_rows, size_fields)

    overall_rows: List[Dict[str, object]] = []
    for configuration in configurations:
        optimizer = str(configuration["optimizer"])
        group = [row for row in successful if row.get("optimizer") == optimizer]
        normalized_values = [float(row["normalized_return"]) for row in group]
        success_values = [float(row["eval_greedy_success"]) for row in group]
        normalized_mean, normalized_std = mean_std(normalized_values)
        success_mean, success_std = mean_std(success_values)
        overall_rows.append(
            {
                "optimizer": optimizer,
                "completed_runs": len(group),
                "mean_normalized_return": normalized_mean,
                "std_normalized_return": normalized_std,
                "mean_greedy_success": success_mean,
                "std_greedy_success": success_std,
                "mean_sps": mean_std([float(row["sps"]) for row in group])[0],
            }
        )
    overall_fields = list(overall_rows[0].keys()) if overall_rows else []
    write_csv(output_root / "overall_summary.csv", overall_rows, overall_fields)

    by_key = {
        (str(row["optimizer"]), int(row["deepsea_size"]), int(row["seed"])): row
        for row in successful
    }
    paired_rows: List[Dict[str, object]] = []
    if any(str(configuration["optimizer"]) == "Adam" for configuration in configurations):
        for configuration in configurations:
            contender = str(configuration["optimizer"])
            if contender == "Adam":
                continue
            for size in sizes:
                for seed in seeds:
                    baseline = by_key.get(("Adam", size, seed))
                    comparison = by_key.get((contender, size, seed))
                    if baseline is None or comparison is None:
                        continue
                    paired_rows.append(
                        {
                            "contender": contender,
                            "baseline": "Adam",
                            "deepsea_size": size,
                            "seed": seed,
                            "normalized_return_delta": float(
                                comparison["normalized_return"]
                            )
                            - float(baseline["normalized_return"]),
                            "success_delta": float(comparison["eval_greedy_success"])
                            - float(baseline["eval_greedy_success"]),
                            "action_accuracy_delta": float(
                                comparison["eval_optimal_action_accuracy"]
                            )
                            - float(baseline["eval_optimal_action_accuracy"]),
                        }
                    )
    paired_fields = list(paired_rows[0].keys()) if paired_rows else [
        "contender",
        "baseline",
        "deepsea_size",
        "seed",
        "normalized_return_delta",
        "success_delta",
        "action_accuracy_delta",
    ]
    write_csv(output_root / "paired_deltas.csv", paired_rows, paired_fields)

    failures = [row for row in run_rows if row.get("status") not in {"ok", "dry_run"}]
    print(json.dumps(overall_rows, indent=2, sort_keys=True))
    print(f"Wrote evaluation tables to {output_root}")
    if failures:
        raise SystemExit(f"{len(failures)} evaluation run(s) failed; see run_results.csv")


if __name__ == "__main__":
    main()

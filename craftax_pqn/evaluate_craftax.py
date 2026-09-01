#!/usr/bin/env python3
"""Evaluate tuned Adam and Muon Craftax PQN configurations on matched seeds."""

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
        parse_envs,
        read_json,
        run_training,
        safe_slug,
    )
except ImportError:
    from launch_utils import (  # type: ignore
        Device,
        normalized_return,
        parse_devices,
        parse_envs,
        read_json,
        run_training,
        safe_slug,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = REPO_ROOT / "cleanrl" / "pqn_craftax.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--envs", default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--output-root", default="logs/craftax/evaluation_1b")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000_000)
    parser.add_argument("--eval-steps", type=int, default=10_000)
    parser.add_argument("--eval-num-envs", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--matrix-diagnostics-interval", type=int, default=100)
    parser.add_argument(
        "--matrix-diagnostics-power-iterations", type=int, default=8
    )
    renorm_group = parser.add_mutually_exclusive_group()
    renorm_group.add_argument(
        "--scale-batch-renorm-warmup",
        dest="scale_batch_renorm_warmup",
        action="store_true",
    )
    renorm_group.add_argument(
        "--no-scale-batch-renorm-warmup",
        dest="scale_batch_renorm_warmup",
        action="store_false",
    )
    parser.set_defaults(scale_batch_renorm_warmup=True)
    parser.add_argument("--cpus-per-run", type=int, default=4)
    track_group = parser.add_mutually_exclusive_group()
    track_group.add_argument("--track", dest="track", action="store_true")
    track_group.add_argument("--no-track", dest="track", action="store_false")
    parser.set_defaults(track=True)
    parser.add_argument(
        "--wandb-project-name", default="cleanRL-craftax-pqn-1b"
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="craftax-pqn-final-evaluation")
    parser.add_argument(
        "--sequential-configs",
        action="store_true",
        help="Finish every seed for one optimizer before starting the next.",
    )
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
        if (
            payload.get("algorithm") != "discrete_pqn"
            or payload.get("environment") != "Craftax"
        ):
            raise ValueError(f"{path} is not a Craftax discrete-PQN tuning result")
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
                "envs": [str(env_id) for env_id in payload.get("envs", [])],
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


def scale_batch_renorm_warmup(
    params: Mapping[str, object], requested_total_timesteps: int
) -> int:
    """Preserve the tuning run's Batch Renorm fraction in optimizer steps."""
    tuning_total_timesteps = int(params["total-timesteps"])
    tuning_warmup_steps = int(params["batch-renorm-warmup-steps"])
    num_envs = int(params["num-envs"])
    num_steps = int(params["num-steps"])
    update_epochs = int(params["update-epochs"])
    num_minibatches = int(params["num-minibatches"])
    rollout_size = num_envs * num_steps
    tuning_iterations = tuning_total_timesteps // rollout_size
    requested_iterations = requested_total_timesteps // rollout_size
    tuning_optimizer_steps = tuning_iterations * update_epochs * num_minibatches
    requested_optimizer_steps = (
        requested_iterations * update_epochs * num_minibatches
    )
    if tuning_optimizer_steps < 1 or requested_optimizer_steps < 1:
        raise ValueError("Both tuning and requested runs need at least one rollout")
    if tuning_warmup_steps == 0:
        return 0
    warmup_fraction = tuning_warmup_steps / tuning_optimizer_steps
    return max(1, round(warmup_fraction * requested_optimizer_steps))


def summary_matches_request(
    summary: Mapping[str, object],
    params: Mapping[str, object],
    env_id: str,
    seed: int,
    *,
    require_wandb: bool,
) -> bool:
    if summary.get("status") != "ok":
        return False
    if summary.get("env_id") != env_id or int(summary.get("seed", -1)) != seed:
        return False
    saved_config = summary.get("config")
    if not isinstance(saved_config, dict):
        return False
    if require_wandb and summary.get("wandb_uploaded") is not True:
        return False
    for key, requested in params.items():
        normalized_key = str(key).replace("-", "_")
        if (
            normalized_key not in saved_config
            or saved_config[normalized_key] != requested
        ):
            return False
    return True


def main() -> None:
    args = parse_args()
    if args.weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative")
    if args.matrix_diagnostics_interval < 1:
        raise ValueError("matrix_diagnostics_interval must be positive")
    if args.matrix_diagnostics_power_iterations < 1:
        raise ValueError("matrix_diagnostics_power_iterations must be positive")
    if args.num_seeds < 1:
        raise ValueError("num_seeds must be positive")
    configurations = load_configs(args.configs)
    for configuration in configurations:
        params = dict(configuration["params"])
        source_total_timesteps = int(params["total-timesteps"])
        source_warmup_steps = int(params["batch-renorm-warmup-steps"])
        source_optimizer_steps = (
            source_total_timesteps
            // (int(params["num-envs"]) * int(params["num-steps"]))
            * int(params["update-epochs"])
            * int(params["num-minibatches"])
        )
        configuration["tuning_batch_renorm_warmup_steps"] = source_warmup_steps
        configuration["tuning_optimizer_steps"] = source_optimizer_steps
        configuration["tuning_batch_renorm_warmup_fraction"] = (
            source_warmup_steps / source_optimizer_steps
        )
        if args.total_timesteps is not None:
            if args.scale_batch_renorm_warmup:
                params["batch-renorm-warmup-steps"] = (
                    scale_batch_renorm_warmup(params, args.total_timesteps)
                )
            params["total-timesteps"] = args.total_timesteps
        params["weight-decay"] = args.weight_decay
        params["matrix-diagnostics-interval"] = (
            args.matrix_diagnostics_interval
        )
        params["matrix-diagnostics-power-iterations"] = (
            args.matrix_diagnostics_power_iterations
        )
        params["eval-steps"] = args.eval_steps
        params["eval-num-envs"] = args.eval_num_envs
        configuration["params"] = params
    devices = parse_devices(args.gpus)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    if args.envs:
        envs = parse_envs(args.envs)
    else:
        envs = sorted(
            {
                str(env_id)
                for configuration in configurations
                for env_id in configuration["envs"]
            }
        )
    if not envs:
        raise ValueError(
            "No Craftax environments were supplied or found in the configs"
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "resolved_configs.json").open("w", encoding="utf-8") as handle:
        json.dump(configurations, handle, indent=2, sort_keys=True)
    for configuration in configurations:
        print(
            "[config] "
            f"optimizer={configuration['optimizer']} "
            f"best_trial={configuration['best_trial_number']} "
            f"total_timesteps={configuration['params']['total-timesteps']} "
            "batch_renorm_warmup_steps="
            f"{configuration['params']['batch-renorm-warmup-steps']} "
            f"weight_decay={configuration['params']['weight-decay']}"
        )
    tasks = [
        {"configuration": configuration, "env_id": env_id, "seed": seed}
        for configuration in configurations
        for env_id in envs
        for seed in seeds
    ]

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
            run_dir = (
                output_root
                / "runs"
                / safe_slug(optimizer)
                / safe_slug(env_id)
                / f"seed_{seed}"
            )
            summary_path = run_dir / "summary.json"
            if (
                summary_path.exists()
                and not args.overwrite
                and not args.dry_run
                and summary_matches_request(
                    read_json(summary_path),
                    params,
                    env_id,
                    seed,
                    require_wandb=args.track,
                )
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
                    env_id=env_id,
                    seed=seed,
                    device=device,
                    output_dir=run_dir,
                    log_path=output_root
                    / "launcher_logs"
                    / f"{safe_slug(optimizer)}__{safe_slug(env_id)}__seed_{seed}.log",
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

    task_groups = (
        [
            [
                task
                for task in tasks
                if task["configuration"] is configuration
            ]
            for configuration in configurations
        ]
        if args.sequential_configs
        else [tasks]
    )
    results: List[Dict[str, object]] = []
    for group_index, task_group in enumerate(task_groups, start=1):
        queues = distribute(task_group, devices)
        if args.sequential_configs:
            optimizer = task_group[0]["configuration"]["optimizer"]
            print(
                f"[phase {group_index}/{len(task_groups)}] "
                f"starting optimizer={optimizer} runs={len(task_group)}"
            )
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(run_queue, device, queue)
                for device, queue in queues.items()
            ]
            for future in as_completed(futures):
                results.extend(future.result())
    results.sort(
        key=lambda item: (
            str(item.get("optimizer")),
            str(item.get("env_id")),
            int(item.get("seed", -1)),
        )
    )

    run_rows: List[Dict[str, object]] = []
    for result in results:
        row = dict(result)
        if result.get("status") == "ok":
            row["normalized_return"] = normalized_return(
                float(result["eval_greedy_return"]), str(result["env_id"])
            )
        run_rows.append(row)
    run_fields = [
        "optimizer",
        "env_id",
        "seed",
        "status",
        "resumed",
        "eval_greedy_return",
        "normalized_return",
        "eval_greedy_return_std",
        "eval_reward_per_1000_steps",
        "eval_episodes",
        "eval_mean_episode_length",
        "eval_partial_return_fallback",
        "sps",
        "elapsed_seconds",
        "final_diagnostics",
        "wandb_uploaded",
        "wandb_run_url",
        "output_dir",
        "log_path",
        "config_path",
        "error",
    ]
    write_csv(output_root / "run_results.csv", run_rows, run_fields)

    successful = [row for row in run_rows if row.get("status") == "ok"]
    environment_rows: List[Dict[str, object]] = []
    for configuration in configurations:
        optimizer = str(configuration["optimizer"])
        for env_id in envs:
            group = [
                row
                for row in successful
                if row.get("optimizer") == optimizer and row.get("env_id") == env_id
            ]
            returns = [float(row["eval_greedy_return"]) for row in group]
            normalized = [float(row["normalized_return"]) for row in group]
            reward_rates = [float(row["eval_reward_per_1000_steps"]) for row in group]
            mean_return, std_return = mean_std(returns)
            mean_normalized, std_normalized = mean_std(normalized)
            mean_rate, std_rate = mean_std(reward_rates)
            environment_rows.append(
                {
                    "optimizer": optimizer,
                    "env_id": env_id,
                    "completed_seeds": len(group),
                    "mean_greedy_return": mean_return,
                    "std_greedy_return": std_return,
                    "mean_normalized_return": mean_normalized,
                    "std_normalized_return": std_normalized,
                    "mean_reward_per_1000_steps": mean_rate,
                    "std_reward_per_1000_steps": std_rate,
                }
            )
    write_csv(
        output_root / "environment_summary.csv",
        environment_rows,
        list(environment_rows[0].keys()) if environment_rows else [],
    )

    overall_rows: List[Dict[str, object]] = []
    for configuration in configurations:
        optimizer = str(configuration["optimizer"])
        group = [row for row in successful if row.get("optimizer") == optimizer]
        normalized_mean, normalized_std = mean_std(
            [float(row["normalized_return"]) for row in group]
        )
        overall_rows.append(
            {
                "optimizer": optimizer,
                "completed_runs": len(group),
                "mean_normalized_return": normalized_mean,
                "std_normalized_return": normalized_std,
                "mean_reward_per_1000_steps": mean_std(
                    [float(row["eval_reward_per_1000_steps"]) for row in group]
                )[0],
                "mean_sps": mean_std([float(row["sps"]) for row in group])[0],
            }
        )
    write_csv(
        output_root / "overall_summary.csv",
        overall_rows,
        list(overall_rows[0].keys()) if overall_rows else [],
    )

    by_key = {
        (str(row["optimizer"]), str(row["env_id"]), int(row["seed"])): row
        for row in successful
    }
    paired_rows: List[Dict[str, object]] = []
    if any(
        str(configuration["optimizer"]) == "Adam" for configuration in configurations
    ):
        for configuration in configurations:
            contender = str(configuration["optimizer"])
            if contender == "Adam":
                continue
            for env_id in envs:
                for seed in seeds:
                    baseline = by_key.get(("Adam", env_id, seed))
                    comparison = by_key.get((contender, env_id, seed))
                    if baseline is None or comparison is None:
                        continue
                    paired_rows.append(
                        {
                            "contender": contender,
                            "baseline": "Adam",
                            "env_id": env_id,
                            "seed": seed,
                            "normalized_return_delta": float(
                                comparison["normalized_return"]
                            )
                            - float(baseline["normalized_return"]),
                            "reward_per_1000_steps_delta": float(
                                comparison["eval_reward_per_1000_steps"]
                            )
                            - float(baseline["eval_reward_per_1000_steps"]),
                        }
                    )
    paired_fields = (
        list(paired_rows[0].keys())
        if paired_rows
        else [
            "contender",
            "baseline",
            "env_id",
            "seed",
            "normalized_return_delta",
            "reward_per_1000_steps_delta",
        ]
    )
    write_csv(output_root / "paired_deltas.csv", paired_rows, paired_fields)

    failures = [row for row in run_rows if row.get("status") not in {"ok", "dry_run"}]
    print(json.dumps(overall_rows, indent=2, sort_keys=True))
    print(f"Wrote evaluation tables to {output_root}")
    if failures:
        raise SystemExit(
            f"{len(failures)} evaluation run(s) failed; see run_results.csv"
        )


if __name__ == "__main__":
    main()

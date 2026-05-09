#!/usr/bin/env python3
"""
eval_best_hparams_from_summary_envpool.py

Evaluate the best hyperparameters from an Optuna summary .txt file.

Important:
- This script does NOT read the Optuna SQLite database.
- It parses the text summary written by write_optuna_summary_file().
- This avoids contamination from older trials in a reused Optuna study DB.

It launches fresh CleanRL training/evaluation runs across more Atari games and
more seeds, using the parsed best hyperparameters.

Example:
  python eval_best_hparams_from_summary_envpool.py \
    --script cleanrl/ppo_atari_envpool.py \
    --best-summary-file tuner_logs/atari10_envpool_RLMuon/optuna_summaries/atari10_envpool_RLMuon__RL_Muon_test.txt \
    --optimizer RLMuon \
    --gpus 0,1,2,3,4,5,6,7 \
    --num-seeds 10 \
    --wandb-project-name cleanrl-atari-best-hparam-eval \
    --wandb-tag RLMuon_best_eval
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Larger evaluation set than your tuning set.
# These should work with EnvPool/Gymnasium Atari v5 if your environment has the ROMs.
ATARI_EVAL_GAMES: List[str] = [
    "Alien-v5",
    "Amidar-v5",
    "Assault-v5",
    "Asterix-v5",
    "BankHeist-v5",
    "BattleZone-v5",
    "BeamRider-v5",
    "Bowling-v5",
    "Boxing-v5",
    "Breakout-v5",
    "ChopperCommand-v5",
    "CrazyClimber-v5",
    "DemonAttack-v5",
    "Enduro-v5",
    "Freeway-v5",
    "Frostbite-v5",
    "Gopher-v5",
    "Gravitar-v5",
    "Kangaroo-v5",
    "Krull-v5",
    "KungFuMaster-v5",
    "MsPacman-v5",
    "Pong-v5",
    "PrivateEye-v5",
    "Qbert-v5",
    "RoadRunner-v5",
    "Seaquest-v5",
    "SpaceInvaders-v5",
    "UpNDown-v5",
    "VideoPinball-v5",
]


# Original tuning set, useful for sanity checking.
TUNING_GAMES: List[str] = [
    "Pong-v5",
    "Breakout-v5",
    "Freeway-v5",
    "Enduro-v5",
    "Seaquest-v5",
    "SpaceInvaders-v5",
    "MsPacman-v5",
    "Assault-v5",
]


def safe_slug(x) -> str:
    """
    Make a string safe for filenames.
    """
    if x is None:
        return "none"

    x = str(x).strip()
    if not x:
        return "none"

    x = re.sub(r"[^A-Za-z0-9_.=-]+", "_", x)
    return x[:140]


def parse_scalar_value(x: str):
    """
    Parse a scalar value from the summary text file.

    Handles:
      int:    5
      float:  0.001
      sci:    1e-4
      string: fallback
    """
    x = x.strip()

    try:
        if "." not in x and "e" not in x.lower():
            return int(x)
    except ValueError:
        pass

    try:
        return float(x)
    except ValueError:
        return x


def load_best_params_from_summary(summary_path: str) -> Dict[str, object]:
    """
    Parse the best hyperparameters from the summary file produced by
    write_optuna_summary_file().

    It reads exactly this section:

      BEST HYPERPARAMETERS / OPTIMAL VARIABLES
      ----------------------------------------------------------------------------------------------------
      lr: ...
      ent_coef: ...
      update_epochs: ...
      momentum: ...

    and stops at the blank line after the section.
    """
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find best-summary-file: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()

    section_name = "BEST HYPERPARAMETERS / OPTIMAL VARIABLES"
    in_section = False
    params: Dict[str, object] = {}

    for raw_line in lines:
        line = raw_line.strip()

        if line == section_name:
            in_section = True
            continue

        if not in_section:
            continue

        # Skip dashed separator.
        if line and set(line) == {"-"}:
            continue

        # The summary writer puts a blank line after the best-param section.
        if line == "":
            break

        if line == "(none)":
            break

        # Defensive stop in case the format changes.
        if line.isupper() and ":" not in line:
            break

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key:
            params[key] = parse_scalar_value(value)

    required = ["lr", "ent_coef", "update_epochs", "momentum"]
    missing = [k for k in required if k not in params]

    if missing:
        raise RuntimeError(
            f"Summary file did not contain required best params: {missing}\n"
            f"summary_path={path}\n"
            f"parsed_params={params}"
        )

    return params


def parse_run_config_from_summary(summary_path: str) -> Dict[str, str]:
    """
    Optional helper: parse RUN CONFIGURATION ARGS from the summary file.

    This is only used for printing/sanity checking. The actual evaluated
    hyperparameters come from BEST HYPERPARAMETERS / OPTIMAL VARIABLES.
    """
    path = Path(summary_path)
    lines = path.read_text(encoding="utf-8").splitlines()

    section_name = "RUN CONFIGURATION ARGS"
    in_section = False
    args: Dict[str, str] = {}

    for raw_line in lines:
        line = raw_line.strip()

        if line == section_name:
            in_section = True
            continue

        if not in_section:
            continue

        if line and set(line) == {"-"}:
            continue

        if line == "":
            break

        if line.isupper() and ":" not in line:
            break

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        args[key.strip()] = value.strip()

    return args


def best_params_to_cleanrl_flags(
    *,
    best_params: Dict[str, object],
    optimizer: str,
    total_timesteps: int,
    num_envs: int,
    num_steps: int,
) -> Dict[str, object]:
    """
    Convert parsed Optuna summary params into CleanRL CLI flags.

    Summary names:
      lr
      ent_coef
      update_epochs
      momentum

    CleanRL CLI flags:
      --learning-rate
      --ent-coef
      --update-epochs
      --momentum
    """
    return {
        "learning-rate": float(best_params["lr"]),
        "ent-coef": float(best_params["ent_coef"]),
        "update-epochs": int(best_params["update_epochs"]),
        "momentum": float(best_params["momentum"]),

        # Fixed eval settings.
        "num-envs": int(num_envs),
        "num-steps": int(num_steps),
        "total-timesteps": int(total_timesteps),

        # Fixed optimizer choice.
        "optimizer": optimizer,
    }


def params_to_argv(params: Dict[str, object]) -> List[str]:
    """
    Convert a dict of CleanRL args into CLI flags.

    Example:
      {"learning-rate": 3e-4, "track": True}
    becomes:
      ["--learning-rate", "0.0003", "--track"]
    """
    argv: List[str] = []

    for key, value in params.items():
        flag = f"--{key}"

        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue

        if value is None:
            continue

        argv.extend([flag, str(value)])

    return argv


def parse_csv_list(x: Optional[str]) -> Optional[List[str]]:
    if x is None:
        return None
    items = [item.strip() for item in x.split(",") if item.strip()]
    return items if items else None


def build_tasks(games: List[str], seeds: List[int]) -> List[Tuple[str, int]]:
    return [(env_id, seed) for env_id in games for seed in seeds]


def split_tasks_across_gpus(
    tasks: List[Tuple[str, int]],
    gpus: List[int],
) -> Dict[int, List[Tuple[str, int]]]:
    """
    Assign tasks round-robin to fixed GPU queues.

    This avoids the subtle oversubscription bug where a GPU can receive a new
    task while a previous task on that same GPU is still running.
    """
    by_gpu: Dict[int, List[Tuple[str, int]]] = {gpu: [] for gpu in gpus}

    for idx, task in enumerate(tasks):
        gpu = gpus[idx % len(gpus)]
        by_gpu[gpu].append(task)

    return by_gpu


def run_one_cleanrl_subprocess(
    *,
    script: str,
    env_id: str,
    seed: int,
    gpu: int,
    cleanrl_params: Dict[str, object],
    wandb_project_name: str,
    wandb_tag: Optional[str],
    wandb_entity: Optional[str],
    logs_root: str,
    cpus_per_run: int,
    extra_args: List[str],
    dry_run: bool,
) -> Dict[str, object]:
    """
    Launch one CleanRL run as a subprocess on exactly one visible GPU.
    """
    Path(logs_root).mkdir(parents=True, exist_ok=True)

    safe_env = safe_slug(env_id)
    safe_tag = safe_slug(wandb_tag)
    log_path = Path(logs_root) / f"{safe_env}__seed{seed}__gpu{gpu}__{safe_tag}.log"

    argv = [
        sys.executable,
        "-u",
        script,
        "--env-id",
        env_id,
        "--seed",
        str(seed),
        "--track",
        "--wandb-project-name",
        wandb_project_name,
    ]

    if wandb_entity:
        argv.extend(["--wandb-entity", wandb_entity])

    if wandb_tag:
        argv.extend(["--wandb-tag", wandb_tag])

    argv.extend(params_to_argv(cleanrl_params))
    argv.extend(extra_args)

    env = os.environ.copy()

    # Each subprocess sees only one physical GPU.
    # Inside the subprocess, cuda:0 maps to this physical GPU.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Avoid CPU oversubscription.
    env["OMP_NUM_THREADS"] = str(cpus_per_run)
    env["MKL_NUM_THREADS"] = str(cpus_per_run)
    env["OPENBLAS_NUM_THREADS"] = str(cpus_per_run)
    env["NUMEXPR_NUM_THREADS"] = str(cpus_per_run)

    # Make W&B less fragile under subprocess-heavy jobs.
    env.setdefault("WANDB_START_METHOD", "thread")
    env.setdefault("WANDB__SERVICE_WAIT", "300")

    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd_str = " ".join(shlex.quote(x) for x in argv)

    print("=" * 100, flush=True)
    print(f"[launch] env={env_id} seed={seed} gpu={gpu}", flush=True)
    print(f"[launch] log={log_path}", flush=True)
    print(f"[launch] cmd={cmd_str}", flush=True)

    if dry_run:
        return {
            "env_id": env_id,
            "seed": seed,
            "gpu": gpu,
            "returncode": 0,
            "elapsed_sec": 0.0,
            "log_path": str(log_path),
            "cmd": cmd_str,
            "dry_run": True,
        }

    start = time.time()

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[cmd] {cmd_str}\n")
        f.write(f"[env] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
        f.write(f"[env] OMP_NUM_THREADS={env['OMP_NUM_THREADS']}\n")
        f.write(f"[env] MKL_NUM_THREADS={env['MKL_NUM_THREADS']}\n")
        f.flush()

        proc = subprocess.run(
            argv,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = time.time() - start

    return {
        "env_id": env_id,
        "seed": seed,
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
        "cmd": cmd_str,
        "dry_run": False,
    }


def run_gpu_queue(
    *,
    gpu: int,
    tasks: List[Tuple[str, int]],
    script: str,
    cleanrl_params: Dict[str, object],
    wandb_project_name: str,
    wandb_tag: Optional[str],
    wandb_entity: Optional[str],
    logs_root: str,
    cpus_per_run: int,
    extra_args: List[str],
    dry_run: bool,
) -> List[Dict[str, object]]:
    """
    Run all tasks assigned to one GPU sequentially.

    One thread controls one GPU. This gives parallelism across GPUs without
    letting two subprocesses accidentally use the same GPU at once.
    """
    results: List[Dict[str, object]] = []

    print(f"[gpu-worker] gpu={gpu} assigned {len(tasks)} tasks", flush=True)

    for env_id, seed in tasks:
        result = run_one_cleanrl_subprocess(
            script=script,
            env_id=env_id,
            seed=seed,
            gpu=gpu,
            cleanrl_params=cleanrl_params,
            wandb_project_name=wandb_project_name,
            wandb_tag=wandb_tag,
            wandb_entity=wandb_entity,
            logs_root=logs_root,
            cpus_per_run=cpus_per_run,
            extra_args=extra_args,
            dry_run=dry_run,
        )

        results.append(result)

        status = "OK" if result["returncode"] == 0 else "FAIL"
        print(
            f"[done:{status}] gpu={gpu} env={env_id} seed={seed} "
            f"returncode={result['returncode']} elapsed_sec={result['elapsed_sec']:.1f} "
            f"log={result['log_path']}",
            flush=True,
        )

        # Continue even if one game fails. At the end, the launcher exits nonzero
        # if there were failures.
        if result["returncode"] != 0:
            print(
                f"[warn] run failed but continuing queue: gpu={gpu} env={env_id} seed={seed}",
                flush=True,
            )

    return results


def write_results_csv(results: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "env_id",
        "seed",
        "gpu",
        "returncode",
        "elapsed_sec",
        "log_path",
        "dry_run",
        "cmd",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in sorted(results, key=lambda x: (str(x["env_id"]), int(x["seed"]))):
            writer.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--script", type=str, default="cleanrl/ppo_atari_envpool.py")
    p.add_argument(
        "--best-summary-file",
        type=str,
        required=True,
        help="Path to Optuna summary .txt file. Best hparams are parsed from this file.",
    )
    p.add_argument(
        "--optimizer",
        type=str,
        required=True,
        help="Optimizer string to pass to CleanRL, e.g. Adam, Muon, RLMuon, NorMuon, AdaMuon.",
    )

    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=1)

    p.add_argument("--game-set", type=str, default="eval30", choices=["eval30", "tuning8"])
    p.add_argument(
        "--games",
        type=str,
        default=None,
        help="Optional comma-separated game override, e.g. Pong-v5,Breakout-v5,Qbert-v5",
    )

    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=256)

    p.add_argument("--wandb-project-name", type=str, default="cleanrl-atari-best-hparam-eval")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tag", type=str, default="best-hparam-eval")

    p.add_argument("--logs-root", type=str, default="eval_logs")
    p.add_argument("--summary-csv", type=str, default=None)

    p.add_argument(
        "--cpus-per-run",
        type=int,
        default=8,
        help="CPU threads assigned to each single-GPU subprocess.",
    )

    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help=(
            "Extra arg passed through literally to CleanRL script. "
            "For flags beginning with --, use equals syntax, e.g. "
            "--extra-arg=--torch-deterministic=False"
        ),
    )

    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("No GPUs provided. Use --gpus 0,1,2,...")

    game_override = parse_csv_list(args.games)
    if game_override is not None:
        games = game_override
    elif args.game_set == "tuning8":
        games = TUNING_GAMES
    else:
        games = ATARI_EVAL_GAMES

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    print("=" * 100)
    print("[eval] loading best hyperparameters from summary file")
    print(f"[eval] best_summary_file={args.best_summary_file}")

    best_params = load_best_params_from_summary(args.best_summary_file)
    run_config = parse_run_config_from_summary(args.best_summary_file)

    print("[eval] parsed best params")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    if run_config:
        print("[eval] summary-file run config sanity info")
        for key in ["study_name", "wandb_tag", "optimizer", "metric", "metric_window", "trials", "seeds"]:
            if key in run_config:
                print(f"  {key}: {run_config[key]}")

    cleanrl_params = best_params_to_cleanrl_flags(
        best_params=best_params,
        optimizer=args.optimizer,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
    )

    print("[eval] CleanRL flags that will be used")
    for k, v in cleanrl_params.items():
        print(f"  --{k} {v}")

    tasks = build_tasks(games, seeds)
    tasks_by_gpu = split_tasks_across_gpus(tasks, gpus)

    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else Path(args.logs_root) / f"{safe_slug(args.wandb_tag)}__eval_summary.csv"
    )

    print("=" * 100)
    print(f"[eval] games: {len(games)}")
    print(f"[eval] seeds per game: {len(seeds)}")
    print(f"[eval] total runs: {len(tasks)}")
    print(f"[eval] GPUs: {gpus}")
    print(f"[eval] max parallel runs: {len(gpus)}")
    print(f"[eval] logs_root: {args.logs_root}")
    print(f"[eval] summary_csv: {summary_csv}")
    print(f"[eval] wandb_project_name: {args.wandb_project_name}")
    print(f"[eval] wandb_tag: {args.wandb_tag}")
    print("=" * 100)

    all_results: List[Dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []

        for gpu in gpus:
            futures.append(
                executor.submit(
                    run_gpu_queue,
                    gpu=gpu,
                    tasks=tasks_by_gpu[gpu],
                    script=args.script,
                    cleanrl_params=cleanrl_params,
                    wandb_project_name=args.wandb_project_name,
                    wandb_tag=args.wandb_tag,
                    wandb_entity=args.wandb_entity,
                    logs_root=args.logs_root,
                    cpus_per_run=args.cpus_per_run,
                    extra_args=args.extra_arg,
                    dry_run=args.dry_run,
                )
            )

        for fut in as_completed(futures):
            gpu_results = fut.result()
            all_results.extend(gpu_results)

            # Incremental checkpoint of launcher-level results.
            write_results_csv(all_results, summary_csv)

    write_results_csv(all_results, summary_csv)

    failures = [r for r in all_results if r.get("returncode") != 0]

    print("=" * 100)
    print("[eval] finished")
    print(f"[eval] total runs: {len(all_results)}")
    print(f"[eval] failures:   {len(failures)}")
    print(f"[eval] summary_csv: {summary_csv}")

    if failures:
        print("[eval] failed runs:")
        for r in failures:
            print(
                f"  env={r.get('env_id')} seed={r.get('seed')} "
                f"gpu={r.get('gpu')} log={r.get('log_path')}"
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
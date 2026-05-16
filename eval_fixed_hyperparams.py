#!/usr/bin/env python3
"""
eval_fixed_hparams_envpool.py

Run fixed-hyperparameter CleanRL Atari EnvPool evaluations across many games,
seeds, and optimizers.

No Optuna.
No best-summary parsing.
No study DB.

Designed for comparing optimizers under the same PPO objective and same fixed
hyperparameters.

Default fixed params:
  lr                = 0.00075
  aux_learning_rate = 0.00075
  ent_coef          = 0.01
  momentum          = 0.95
  update_epochs     = 5
  num_envs          = 32
  num_steps         = 256
  total_timesteps   = 10_000_000
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    if x is None:
        return "none"
    x = str(x).strip()
    if not x:
        return "none"
    x = re.sub(r"[^A-Za-z0-9_.=-]+", "_", x)
    return x[:140]


def parse_csv_list(x: Optional[str]) -> Optional[List[str]]:
    if x is None:
        return None
    items = [item.strip() for item in x.split(",") if item.strip()]
    return items if items else None


def setup_logging(logs_root: str, tag: str) -> Path:
    Path(logs_root).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(logs_root) / f"launcher__{safe_slug(tag)}__{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    root.addHandler(fh)
    root.addHandler(sh)

    logging.info("=" * 100)
    logging.info("launcher log path: %s", log_path)
    logging.info("=" * 100)

    return log_path


def fixed_cleanrl_params(
    *,
    optimizer: str,
    learning_rate: float,
    aux_learning_rate: float,
    ent_coef: float,
    momentum: float,
    update_epochs: int,
    num_envs: int,
    num_steps: int,
    total_timesteps: int,
) -> Dict[str, object]:
    return {
        "learning-rate": float(learning_rate),
        "aux-learning-rate": float(aux_learning_rate),
        "ent-coef": float(ent_coef),
        "update-epochs": int(update_epochs),
        "momentum": float(momentum),
        "num-envs": int(num_envs),
        "num-steps": int(num_steps),
        "total-timesteps": int(total_timesteps),
        "optimizer": optimizer,
    }


def params_to_argv(params: Dict[str, object]) -> List[str]:
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


def build_tasks(
    *,
    optimizers: List[str],
    games: List[str],
    seeds: List[int],
) -> List[Tuple[str, str, int]]:
    """
    Returns:
      [(optimizer, env_id, seed), ...]
    """
    return [
        (optimizer, env_id, seed)
        for optimizer in optimizers
        for env_id in games
        for seed in seeds
    ]


def split_tasks_across_gpus(
    tasks: List[Tuple[str, str, int]],
    gpus: List[int],
) -> Dict[int, List[Tuple[str, str, int]]]:
    by_gpu: Dict[int, List[Tuple[str, str, int]]] = {gpu: [] for gpu in gpus}

    for idx, task in enumerate(tasks):
        gpu = gpus[idx % len(gpus)]
        by_gpu[gpu].append(task)

    return by_gpu


def tail_file(path: Path, n: int = 80) -> str:
    try:
        if not path.exists():
            return f"(log file does not exist: {path})"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return f"(failed to read tail of {path})\n{traceback.format_exc()}"


def run_one_cleanrl_subprocess(
    *,
    script: str,
    optimizer: str,
    env_id: str,
    seed: int,
    gpu: int,
    cleanrl_params: Dict[str, object],
    wandb_project_name: str,
    wandb_tag_prefix: str,
    wandb_entity: Optional[str],
    logs_root: str,
    cpus_per_run: int,
    extra_args: List[str],
    dry_run: bool,
) -> Dict[str, object]:
    try:
        Path(logs_root).mkdir(parents=True, exist_ok=True)

        optimizer_slug = safe_slug(optimizer)
        env_slug = safe_slug(env_id)

        # One W&B tag per optimizer, so filtering is easy.
        wandb_tag = f"{wandb_tag_prefix}_{optimizer_slug}"

        log_path = (
            Path(logs_root)
            / optimizer_slug
            / f"{env_slug}__seed{seed}__gpu{gpu}__{safe_slug(wandb_tag)}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)

        script_path = Path(script)
        if not script_path.exists():
            raise FileNotFoundError(f"CleanRL script does not exist: {script_path.resolve()}")

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
            "--wandb-tag",
            wandb_tag,
        ]

        if wandb_entity:
            argv.extend(["--wandb-entity", wandb_entity])

        argv.extend(params_to_argv(cleanrl_params))
        argv.extend(extra_args)

        env = os.environ.copy()

        # Each child process sees exactly one physical GPU.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        env["OMP_NUM_THREADS"] = str(cpus_per_run)
        env["MKL_NUM_THREADS"] = str(cpus_per_run)
        env["OPENBLAS_NUM_THREADS"] = str(cpus_per_run)
        env["NUMEXPR_NUM_THREADS"] = str(cpus_per_run)

        env.setdefault("WANDB_START_METHOD", "thread")
        env.setdefault("WANDB__SERVICE_WAIT", "300")

        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        cmd_str = " ".join(shlex.quote(x) for x in argv)

        logging.info("=" * 100)
        logging.info("[launch] optimizer=%s env=%s seed=%s gpu=%s", optimizer, env_id, seed, gpu)
        logging.info("[launch] log=%s", log_path)
        logging.info("[launch] cmd=%s", cmd_str)

        with log_path.open("w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("CLEANRL FIXED-HPARAM SUBPROCESS LAUNCH LOG\n")
            f.write("=" * 100 + "\n")
            f.write(f"optimizer: {optimizer}\n")
            f.write(f"env_id: {env_id}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"physical_gpu: {gpu}\n")
            f.write(f"wandb_project_name: {wandb_project_name}\n")
            f.write(f"wandb_tag: {wandb_tag}\n")
            f.write(f"cwd: {os.getcwd()}\n")
            f.write(f"python: {sys.executable}\n")
            f.write(f"cmd: {cmd_str}\n")
            f.write(f"CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}\n")
            f.write(f"OMP_NUM_THREADS: {env['OMP_NUM_THREADS']}\n")
            f.write(f"MKL_NUM_THREADS: {env['MKL_NUM_THREADS']}\n")
            f.write(f"OPENBLAS_NUM_THREADS: {env['OPENBLAS_NUM_THREADS']}\n")
            f.write(f"NUMEXPR_NUM_THREADS: {env['NUMEXPR_NUM_THREADS']}\n")
            f.write("=" * 100 + "\n\n")
            f.flush()

        if dry_run:
            logging.info("[dry-run] not launching subprocess")
            return {
                "optimizer": optimizer,
                "env_id": env_id,
                "seed": seed,
                "gpu": gpu,
                "returncode": 0,
                "elapsed_sec": 0.0,
                "log_path": str(log_path),
                "wandb_tag": wandb_tag,
                "cmd": cmd_str,
                "dry_run": True,
            }

        start = time.time()

        with log_path.open("a", encoding="utf-8") as f:
            proc = subprocess.run(
                argv,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        elapsed = time.time() - start

        result = {
            "optimizer": optimizer,
            "env_id": env_id,
            "seed": seed,
            "gpu": gpu,
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "log_path": str(log_path),
            "wandb_tag": wandb_tag,
            "cmd": cmd_str,
            "dry_run": False,
        }

        if proc.returncode != 0:
            logging.error(
                "[run failed] optimizer=%s env=%s seed=%s gpu=%s returncode=%s log=%s",
                optimizer,
                env_id,
                seed,
                gpu,
                proc.returncode,
                log_path,
            )
            logging.error("last lines of failed log:\n%s", tail_file(log_path, n=100))
        else:
            logging.info(
                "[run ok] optimizer=%s env=%s seed=%s gpu=%s elapsed_sec=%.1f",
                optimizer,
                env_id,
                seed,
                gpu,
                elapsed,
            )

        return result

    except Exception as e:
        logging.exception(
            "[launcher exception] optimizer=%s env=%s seed=%s gpu=%s error=%r",
            optimizer,
            env_id,
            seed,
            gpu,
            e,
        )
        return {
            "optimizer": optimizer,
            "env_id": env_id,
            "seed": seed,
            "gpu": gpu,
            "returncode": -999,
            "elapsed_sec": 0.0,
            "log_path": "",
            "wandb_tag": "",
            "cmd": "",
            "dry_run": dry_run,
            "error": repr(e),
        }


def run_gpu_queue(
    *,
    gpu: int,
    tasks: List[Tuple[str, str, int]],
    script: str,
    fixed_args,
    wandb_project_name: str,
    wandb_tag_prefix: str,
    wandb_entity: Optional[str],
    logs_root: str,
    cpus_per_run: int,
    extra_args: List[str],
    dry_run: bool,
    fail_fast: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []

    logging.info("[gpu-worker] gpu=%s assigned %d tasks", gpu, len(tasks))

    for idx, (optimizer, env_id, seed) in enumerate(tasks, start=1):
        logging.info(
            "[gpu-worker] gpu=%s starting task %d/%d: optimizer=%s env=%s seed=%s",
            gpu,
            idx,
            len(tasks),
            optimizer,
            env_id,
            seed,
        )

        cleanrl_params = fixed_cleanrl_params(
            optimizer=optimizer,
            learning_rate=fixed_args.learning_rate,
            aux_learning_rate=fixed_args.aux_learning_rate,
            ent_coef=fixed_args.ent_coef,
            momentum=fixed_args.momentum,
            update_epochs=fixed_args.update_epochs,
            num_envs=fixed_args.num_envs,
            num_steps=fixed_args.num_steps,
            total_timesteps=fixed_args.total_timesteps,
        )

        result = run_one_cleanrl_subprocess(
            script=script,
            optimizer=optimizer,
            env_id=env_id,
            seed=seed,
            gpu=gpu,
            cleanrl_params=cleanrl_params,
            wandb_project_name=wandb_project_name,
            wandb_tag_prefix=wandb_tag_prefix,
            wandb_entity=wandb_entity,
            logs_root=logs_root,
            cpus_per_run=cpus_per_run,
            extra_args=extra_args,
            dry_run=dry_run,
        )

        results.append(result)

        status = "OK" if result["returncode"] == 0 else "FAIL"
        logging.info(
            "[done:%s] gpu=%s optimizer=%s env=%s seed=%s returncode=%s elapsed_sec=%s log=%s",
            status,
            gpu,
            optimizer,
            env_id,
            seed,
            result.get("returncode"),
            result.get("elapsed_sec"),
            result.get("log_path"),
        )

        if fail_fast and result["returncode"] != 0:
            logging.error("[fail-fast] stopping GPU queue after failure on gpu=%s", gpu)
            break

    return results


def write_results_csv(results: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "optimizer",
        "env_id",
        "seed",
        "gpu",
        "returncode",
        "elapsed_sec",
        "log_path",
        "wandb_tag",
        "dry_run",
        "cmd",
        "error",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in sorted(
            results,
            key=lambda x: (
                str(x.get("optimizer")),
                str(x.get("env_id")),
                int(x.get("seed", -1)),
            ),
        ):
            writer.writerow({k: r.get(k, "") for k in fields})


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--script", type=str, default="cleanrl/ppo_atari_envpool.py")

    p.add_argument(
        "--optimizers",
        type=str,
        default="Adam,Muon",
        help="Comma-separated optimizer names to evaluate, e.g. Adam,Muon",
    )

    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=1)

    p.add_argument("--game-set", type=str, default="eval30", choices=["eval30", "tuning8"])
    p.add_argument(
        "--games",
        type=str,
        default=None,
        help="Optional comma-separated game override, e.g. Pong-v5,Enduro-v5",
    )

    # Fixed hyperparameters.
    p.add_argument("--learning-rate", type=float, default=0.00075)
    p.add_argument("--aux-learning-rate", type=float, default=0.00075)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--update-epochs", type=int, default=5)

    # Fixed batch/run setup.
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=256)

    p.add_argument("--wandb-project-name", type=str, default="cleanrl-atari-fixed-hparam-eval")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tag-prefix", type=str, default="fixed_lr00075_ent001_10M")

    p.add_argument("--logs-root", type=str, default="eval_logs/fixed_lr00075_ent001_10M")
    p.add_argument("--summary-csv", type=str, default=None)

    p.add_argument("--cpus-per-run", type=int, default=8)

    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help=(
            "Extra arg passed literally to CleanRL. For flags beginning with --, "
            "use equals syntax, e.g. --extra-arg=--torch-deterministic=False"
        ),
    )

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-fast", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    launcher_log = setup_logging(args.logs_root, args.wandb_tag_prefix)

    try:
        logging.info("argv: %s", " ".join(shlex.quote(x) for x in sys.argv))
        logging.info("cwd: %s", os.getcwd())
        logging.info("python executable: %s", sys.executable)
        logging.info("python version: %s", sys.version.replace("\n", " "))
        logging.info("launcher log: %s", launcher_log)

        logging.info("parsed args:")
        for k, v in sorted(vars(args).items()):
            logging.info("  %s: %r", k, v)

        script_path = Path(args.script)
        if not script_path.exists():
            raise FileNotFoundError(f"CleanRL script does not exist: {script_path.resolve()}")

        optimizers = parse_csv_list(args.optimizers)
        if not optimizers:
            raise ValueError("No optimizers provided. Use --optimizers Adam,Muon")

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

        logging.info("selected optimizers (%d): %s", len(optimizers), optimizers)
        logging.info("selected games (%d): %s", len(games), games)
        logging.info("selected seeds (%d): %s", len(seeds), seeds)
        logging.info("selected GPUs (%d): %s", len(gpus), gpus)

        logging.info("fixed hyperparameters:")
        logging.info("  learning_rate:     %s", args.learning_rate)
        logging.info("  aux_learning_rate: %s", args.aux_learning_rate)
        logging.info("  ent_coef:          %s", args.ent_coef)
        logging.info("  momentum:          %s", args.momentum)
        logging.info("  update_epochs:     %s", args.update_epochs)
        logging.info("  num_envs:          %s", args.num_envs)
        logging.info("  num_steps:         %s", args.num_steps)
        logging.info("  total_timesteps:   %s", args.total_timesteps)

        tasks = build_tasks(
            optimizers=optimizers,
            games=games,
            seeds=seeds,
        )
        tasks_by_gpu = split_tasks_across_gpus(tasks, gpus)

        summary_csv = (
            Path(args.summary_csv)
            if args.summary_csv
            else Path(args.logs_root) / f"{safe_slug(args.wandb_tag_prefix)}__fixed_eval_summary.csv"
        )

        logging.info("=" * 100)
        logging.info("total tasks: %d", len(tasks))
        logging.info("max parallel runs: %d", len(gpus))
        logging.info("summary_csv: %s", summary_csv)

        for gpu in gpus:
            logging.info("gpu=%s gets %d tasks", gpu, len(tasks_by_gpu[gpu]))
            logging.info("gpu=%s first tasks: %s", gpu, tasks_by_gpu[gpu][:5])

        logging.info("=" * 100)

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
                        fixed_args=args,
                        wandb_project_name=args.wandb_project_name,
                        wandb_tag_prefix=args.wandb_tag_prefix,
                        wandb_entity=args.wandb_entity,
                        logs_root=args.logs_root,
                        cpus_per_run=args.cpus_per_run,
                        extra_args=args.extra_arg,
                        dry_run=args.dry_run,
                        fail_fast=args.fail_fast,
                    )
                )

            for fut in as_completed(futures):
                try:
                    gpu_results = fut.result()
                    all_results.extend(gpu_results)
                    write_results_csv(all_results, summary_csv)
                    logging.info("wrote incremental summary csv: %s", summary_csv)
                except Exception:
                    logging.exception("GPU worker future crashed")
                    if args.fail_fast:
                        raise

        write_results_csv(all_results, summary_csv)

        failures = [r for r in all_results if r.get("returncode") != 0]

        logging.info("=" * 100)
        logging.info("fixed-hparam evaluation finished")
        logging.info("total results: %d", len(all_results))
        logging.info("failures: %d", len(failures))
        logging.info("summary_csv: %s", summary_csv)
        logging.info("launcher_log: %s", launcher_log)

        if failures:
            logging.error("failed runs:")
            for r in failures:
                logging.error(
                    "  optimizer=%s env=%s seed=%s gpu=%s returncode=%s log=%s error=%s",
                    r.get("optimizer"),
                    r.get("env_id"),
                    r.get("seed"),
                    r.get("gpu"),
                    r.get("returncode"),
                    r.get("log_path"),
                    r.get("error", ""),
                )

            raise SystemExit(1)

    except SystemExit:
        raise
    except Exception:
        logging.exception("fatal launcher crash")
        raise


if __name__ == "__main__":
    main()
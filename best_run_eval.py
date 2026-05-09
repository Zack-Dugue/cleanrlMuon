#!/usr/bin/env python3
"""
eval_best_hparams_from_summary_envpool.py

Evaluate best hyperparameters from an Optuna summary .txt file.

This version has aggressive logging because cluster/subprocess failures can be
otherwise invisible when the job crashes instantly.

Important:
- Does NOT read the Optuna SQLite DB.
- Parses BEST HYPERPARAMETERS / OPTIMAL VARIABLES from the summary .txt file.
- Launches CleanRL runs as subprocesses.
- Assigns each GPU a sequential queue to avoid GPU oversubscription.
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


def setup_logging(logs_root: str, wandb_tag: str) -> Path:
    Path(logs_root).mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(logs_root) / f"launcher__{safe_slug(wandb_tag)}__{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    logging.info("=" * 100)
    logging.info("launcher log path: %s", log_path)
    logging.info("=" * 100)

    return log_path


def parse_scalar_value(x: str):
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
    path = Path(summary_path)
    logging.info("checking best summary file: %s", path.resolve())

    if not path.exists():
        raise FileNotFoundError(f"Could not find best-summary-file: {path}")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    logging.info("summary file exists")
    logging.info("summary file size bytes: %s", path.stat().st_size)
    logging.info("summary file lines: %s", len(lines))

    section_name = "BEST HYPERPARAMETERS / OPTIMAL VARIABLES"
    in_section = False
    params: Dict[str, object] = {}

    for i, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if line == section_name:
            logging.info("found best-param section at line %d", i)
            in_section = True
            continue

        if not in_section:
            continue

        if line and set(line) == {"-"}:
            continue

        if line == "":
            logging.info("end of best-param section at line %d", i)
            break

        if line == "(none)":
            logging.info("best-param section says none")
            break

        if line.isupper() and ":" not in line:
            logging.info("stopping best-param parse at next section line %d: %s", i, line)
            break

        if ":" not in line:
            logging.info("skipping non key/value line %d in best-param section: %s", i, line)
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key:
            parsed = parse_scalar_value(value)
            params[key] = parsed
            logging.info("parsed best param: %s = %r", key, parsed)

    required = ["lr", "ent_coef", "update_epochs", "momentum"]
    missing = [k for k in required if k not in params]

    if missing:
        raise RuntimeError(
            f"Summary file did not contain required best params: {missing}\n"
            f"summary_path={path}\n"
            f"parsed_params={params}\n"
            f"Expected section header exactly: {section_name!r}"
        )

    return params


def parse_run_config_from_summary(summary_path: str) -> Dict[str, str]:
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
    flags = {
        "learning-rate": float(best_params["lr"]),
        "ent-coef": float(best_params["ent_coef"]),
        "update-epochs": int(best_params["update_epochs"]),
        "momentum": float(best_params["momentum"]),
        "num-envs": int(num_envs),
        "num-steps": int(num_steps),
        "total-timesteps": int(total_timesteps),
        "optimizer": optimizer,
    }

    logging.info("converted CleanRL flags:")
    for k, v in flags.items():
        logging.info("  --%s %s", k, v)

    return flags


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
    by_gpu: Dict[int, List[Tuple[str, int]]] = {gpu: [] for gpu in gpus}

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
    try:
        Path(logs_root).mkdir(parents=True, exist_ok=True)

        safe_env = safe_slug(env_id)
        safe_tag = safe_slug(wandb_tag)
        log_path = Path(logs_root) / f"{safe_env}__seed{seed}__gpu{gpu}__{safe_tag}.log"

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
        ]

        if wandb_entity:
            argv.extend(["--wandb-entity", wandb_entity])

        if wandb_tag:
            argv.extend(["--wandb-tag", wandb_tag])

        argv.extend(params_to_argv(cleanrl_params))
        argv.extend(extra_args)

        env = os.environ.copy()
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
        logging.info("[launch] env=%s seed=%s physical_gpu=%s", env_id, seed, gpu)
        logging.info("[launch] log=%s", log_path)
        logging.info("[launch] cwd=%s", os.getcwd())
        logging.info("[launch] python=%s", sys.executable)
        logging.info("[launch] CUDA_VISIBLE_DEVICES for child=%s", env["CUDA_VISIBLE_DEVICES"])
        logging.info("[launch] cmd=%s", cmd_str)

        with log_path.open("w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("CLEANRL SUBPROCESS LAUNCH LOG\n")
            f.write("=" * 100 + "\n")
            f.write(f"env_id: {env_id}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"physical_gpu: {gpu}\n")
            f.write(f"cwd: {os.getcwd()}\n")
            f.write(f"python: {sys.executable}\n")
            f.write(f"cmd: {cmd_str}\n")
            f.write(f"CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}\n")
            f.write(f"OMP_NUM_THREADS: {env['OMP_NUM_THREADS']}\n")
            f.write(f"MKL_NUM_THREADS: {env['MKL_NUM_THREADS']}\n")
            f.write(f"OPENBLAS_NUM_THREADS: {env['OPENBLAS_NUM_THREADS']}\n")
            f.write(f"NUMEXPR_NUM_THREADS: {env['NUMEXPR_NUM_THREADS']}\n")
            f.write(f"WANDB_START_METHOD: {env.get('WANDB_START_METHOD')}\n")
            f.write(f"WANDB__SERVICE_WAIT: {env.get('WANDB__SERVICE_WAIT')}\n")
            f.write("=" * 100 + "\n\n")
            f.flush()

        if dry_run:
            logging.info("[dry-run] not launching subprocess for env=%s seed=%s", env_id, seed)
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
            "env_id": env_id,
            "seed": seed,
            "gpu": gpu,
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "log_path": str(log_path),
            "cmd": cmd_str,
            "dry_run": False,
        }

        if proc.returncode != 0:
            logging.error(
                "[run failed] env=%s seed=%s gpu=%s returncode=%s log=%s",
                env_id,
                seed,
                gpu,
                proc.returncode,
                log_path,
            )
            logging.error("last lines of failed log:\n%s", tail_file(log_path, n=100))
        else:
            logging.info(
                "[run ok] env=%s seed=%s gpu=%s elapsed_sec=%.1f",
                env_id,
                seed,
                gpu,
                elapsed,
            )

        return result

    except Exception as e:
        logging.exception(
            "[launcher exception before/during subprocess] env=%s seed=%s gpu=%s error=%r",
            env_id,
            seed,
            gpu,
            e,
        )
        return {
            "env_id": env_id,
            "seed": seed,
            "gpu": gpu,
            "returncode": -999,
            "elapsed_sec": 0.0,
            "log_path": "",
            "cmd": "",
            "dry_run": dry_run,
            "error": repr(e),
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
    fail_fast: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []

    logging.info("[gpu-worker] gpu=%s assigned %d tasks", gpu, len(tasks))

    for idx, (env_id, seed) in enumerate(tasks, start=1):
        logging.info(
            "[gpu-worker] gpu=%s starting task %d/%d: env=%s seed=%s",
            gpu,
            idx,
            len(tasks),
            env_id,
            seed,
        )

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
        logging.info(
            "[done:%s] gpu=%s env=%s seed=%s returncode=%s elapsed_sec=%s log=%s",
            status,
            gpu,
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
        "env_id",
        "seed",
        "gpu",
        "returncode",
        "elapsed_sec",
        "log_path",
        "dry_run",
        "cmd",
        "error",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in sorted(results, key=lambda x: (str(x.get("env_id")), int(x.get("seed", -1)))):
            writer.writerow({k: r.get(k, "") for k in fields})


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--script", type=str, default="cleanrl/ppo_atari_envpool.py")
    p.add_argument("--best-summary-file", type=str, required=True)
    p.add_argument("--optimizer", type=str, required=True)

    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=1)

    p.add_argument("--game-set", type=str, default="eval30", choices=["eval30", "tuning8"])
    p.add_argument("--games", type=str, default=None)

    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=256)

    p.add_argument("--wandb-project-name", type=str, default="cleanrl-atari-best-hparam-eval")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tag", type=str, default="best-hparam-eval")

    p.add_argument("--logs-root", type=str, default="eval_logs")
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
    launcher_log = setup_logging(args.logs_root, args.wandb_tag)

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
        summary_path = Path(args.best_summary_file)

        logging.info("script path exists: %s -> %s", script_path, script_path.exists())
        logging.info("summary path exists: %s -> %s", summary_path, summary_path.exists())

        if not script_path.exists():
            raise FileNotFoundError(f"CleanRL script does not exist: {script_path.resolve()}")

        if not summary_path.exists():
            raise FileNotFoundError(f"Best summary file does not exist: {summary_path.resolve()}")

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

        logging.info("selected games (%d): %s", len(games), games)
        logging.info("selected seeds (%d): %s", len(seeds), seeds)
        logging.info("selected GPUs (%d): %s", len(gpus), gpus)

        best_params = load_best_params_from_summary(args.best_summary_file)
        run_config = parse_run_config_from_summary(args.best_summary_file)

        logging.info("parsed best params:")
        for k, v in best_params.items():
            logging.info("  %s: %r", k, v)

        if run_config:
            logging.info("summary run config:")
            for k, v in sorted(run_config.items()):
                logging.info("  %s: %s", k, v)

        cleanrl_params = best_params_to_cleanrl_flags(
            best_params=best_params,
            optimizer=args.optimizer,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
        )

        tasks = build_tasks(games, seeds)
        tasks_by_gpu = split_tasks_across_gpus(tasks, gpus)

        logging.info("=" * 100)
        logging.info("total tasks: %d", len(tasks))
        for gpu in gpus:
            logging.info("gpu=%s gets %d tasks", gpu, len(tasks_by_gpu[gpu]))
            preview = tasks_by_gpu[gpu][:5]
            logging.info("gpu=%s first tasks: %s", gpu, preview)
        logging.info("=" * 100)

        summary_csv = (
            Path(args.summary_csv)
            if args.summary_csv
            else Path(args.logs_root) / f"{safe_slug(args.wandb_tag)}__eval_summary.csv"
        )

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
        logging.info("evaluation finished")
        logging.info("total results: %d", len(all_results))
        logging.info("failures: %d", len(failures))
        logging.info("summary_csv: %s", summary_csv)
        logging.info("launcher_log: %s", launcher_log)

        if failures:
            logging.error("failed runs:")
            for r in failures:
                logging.error(
                    "  env=%s seed=%s gpu=%s returncode=%s log=%s error=%s",
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
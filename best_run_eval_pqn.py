#!/usr/bin/env python3
"""
eval_best_pqn_hparams_envpool.py

Evaluate best PQN hyperparameters from a saved tuning summary file.

Supports either:
1. Simple best_hyperparams.txt format:

    study_name: ...
    optimizer: Muon
    ...
    best_params:
      lr: 0.00127
      distance_from_one: 0.1519
      exploration_fraction: 0.1228

2. Longer Optuna summary format with section:

    BEST HYPERPARAMETERS / OPTIMAL VARIABLES
    ------------------------------------------------
    lr: ...
    distance_from_one: ...
    exploration_fraction: ...

This launcher:
- Does NOT read the Optuna SQLite DB.
- Launches CleanRL/PQN runs as subprocesses.
- Assigns each GPU a sequential queue to avoid GPU oversubscription.
- Passes PQN flags:
    --learning-rate
    --q-lambda
    --exploration-fraction
    --num-envs
    --num-steps
    --num-minibatches
    --update-epochs
    --total-timesteps
    --optimizer
    --momentum

No weight decay is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
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

    if x.lower() in {"true", "false"}:
        return x.lower() == "true"

    try:
        if "." not in x and "e" not in x.lower():
            return int(x)
    except ValueError:
        pass

    try:
        return float(x)
    except ValueError:
        return x


def parse_key_value_section(lines: List[str], section_name: str) -> Dict[str, object]:
    """
    Parse long Optuna summary sections like:

        BEST HYPERPARAMETERS / OPTIMAL VARIABLES
        ------------------------------------------------
        lr: 0.001
        distance_from_one: 0.15

    Stops at blank line or next all-caps section.
    """
    in_section = False
    params: Dict[str, object] = {}

    for i, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if line == section_name:
            logging.info("found section %r at line %d", section_name, i)
            in_section = True
            continue

        if not in_section:
            continue

        if line and set(line) == {"-"}:
            continue

        if line == "":
            logging.info("end of section %r at line %d", section_name, i)
            break

        if line == "(none)":
            logging.info("section %r says none", section_name)
            break

        if line.isupper() and ":" not in line:
            logging.info("stopping section parse at next section line %d: %s", i, line)
            break

        if ":" not in line:
            logging.info("skipping non key/value line %d in section: %s", i, line)
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key:
            parsed = parse_scalar_value(value)
            params[key] = parsed
            logging.info("parsed section param: %s = %r", key, parsed)

    return params


def parse_simple_best_params_block(lines: List[str]) -> Dict[str, object]:
    """
    Parse simple file format:

        best_params:
          lr: 0.001
          distance_from_one: 0.15
          exploration_fraction: 0.12
    """
    params: Dict[str, object] = {}
    in_block = False

    for i, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        if stripped == "best_params:":
            logging.info("found simple best_params block at line %d", i)
            in_block = True
            continue

        if not in_block:
            continue

        if stripped == "":
            break

        # Stop if we hit a new top-level key that is not indented.
        if raw_line == raw_line.lstrip() and ":" in raw_line:
            break

        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key:
            parsed = parse_scalar_value(value)
            params[key] = parsed
            logging.info("parsed simple best param: %s = %r", key, parsed)

    return params


def load_best_params_from_summary(summary_path: str) -> Dict[str, object]:
    path = Path(summary_path)
    logging.info("checking best summary file: %s", path.resolve())

    if not path.exists():
        raise FileNotFoundError(f"Could not find best-summary-file: {path}")

    logging.info("summary file exists")
    logging.info("summary file size bytes: %s", path.stat().st_size)

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(payload, dict) and "best_params" in payload:
            params = payload["best_params"]
        elif isinstance(payload, dict):
            params = payload
        else:
            raise RuntimeError(f"Unsupported JSON structure in {path}")

        parsed = {str(k): parse_scalar_value(str(v)) for k, v in params.items()}
        logging.info("parsed JSON best params: %s", parsed)
        return parsed

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    logging.info("summary file lines: %s", len(lines))

    params = parse_key_value_section(
        lines,
        section_name="BEST HYPERPARAMETERS / OPTIMAL VARIABLES",
    )

    if not params:
        params = parse_simple_best_params_block(lines)

    if not params:
        raise RuntimeError(
            f"Could not parse best params from summary file: {path}\n"
            "Expected either a 'BEST HYPERPARAMETERS / OPTIMAL VARIABLES' section "
            "or a simple 'best_params:' block."
        )

    if "lr" not in params and "learning-rate" not in params and "learning_rate" not in params:
        raise RuntimeError(
            f"Best params did not contain lr / learning-rate / learning_rate.\n"
            f"path={path}\n"
            f"parsed_params={params}"
        )

    return params


def parse_run_config_from_summary(summary_path: str) -> Dict[str, str]:
    path = Path(summary_path)

    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {
                k: str(v)
                for k, v in payload.items()
                if k != "best_params"
            }
        except Exception:
            return {}

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


def get_best_float(
    params: Dict[str, object],
    names: List[str],
    default: Optional[float] = None,
) -> float:
    for name in names:
        if name in params:
            return float(params[name])

    if default is not None:
        return float(default)

    raise KeyError(f"Missing required float param. Tried names={names}. params={params}")


def get_best_int(
    params: Dict[str, object],
    names: List[str],
    default: Optional[int] = None,
) -> int:
    for name in names:
        if name in params:
            return int(params[name])

    if default is not None:
        return int(default)

    raise KeyError(f"Missing required int param. Tried names={names}. params={params}")


def derive_q_lambda(best_params: Dict[str, object], default_q_lambda: float) -> float:
    """
    Your PQN tuner stored distance_from_one, not q_lambda:

        distance_from_one = 1 - q_lambda

    So reconstruct q_lambda if needed.
    """
    if "q_lambda" in best_params:
        q_lambda = float(best_params["q_lambda"])
    elif "q-lambda" in best_params:
        q_lambda = float(best_params["q-lambda"])
    elif "lambda" in best_params:
        q_lambda = float(best_params["lambda"])
    elif "distance_from_one" in best_params:
        q_lambda = 1.0 - float(best_params["distance_from_one"])
    else:
        q_lambda = float(default_q_lambda)

    q_lambda = max(0.0, min(1.0, q_lambda))
    return q_lambda


def best_params_to_cleanrl_flags(
    *,
    best_params: Dict[str, object],
    optimizer: str,
    total_timesteps: int,
    num_envs: int,
    num_steps: int,
    num_minibatches: int,
    update_epochs: int,
    momentum: float,
    default_q_lambda: float,
    default_exploration_fraction: float,
    end_e: Optional[float],
) -> Dict[str, object]:
    learning_rate = get_best_float(
        best_params,
        names=["lr", "learning-rate", "learning_rate"],
    )

    q_lambda = derive_q_lambda(best_params, default_q_lambda=default_q_lambda)

    exploration_fraction = get_best_float(
        best_params,
        names=["exploration_fraction", "exploration-fraction"],
        default=default_exploration_fraction,
    )

    update_epochs = get_best_int(
        best_params,
        names=["update_epochs", "update-epochs"],
        default=update_epochs,
    )

    momentum = get_best_float(
        best_params,
        names=["momentum"],
        default=momentum,
    )

    flags: Dict[str, object] = {
        "learning-rate": learning_rate,
        "q-lambda": q_lambda,
        "exploration-fraction": exploration_fraction,
        "num-envs": int(num_envs),
        "num-steps": int(num_steps),
        "num-minibatches": int(num_minibatches),
        "update-epochs": int(update_epochs),
        "total-timesteps": int(total_timesteps),
        "optimizer": optimizer,
        "momentum": float(momentum),
    }

    if end_e is not None:
        flags["end-e"] = float(end_e)

    logging.info("converted PQN CleanRL flags:")
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
    track: bool,
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
        ]

        if track:
            argv.extend([
                "--track",
                "--wandb-project-name",
                wandb_project_name,
            ])

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
    track: bool,
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
            track=track,
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

    p.add_argument("--script", type=str, default="cleanrl/pqn_atari_envpool_wandb.py")
    p.add_argument("--best-summary-file", type=str, required=True)
    p.add_argument("--optimizer", type=str, required=True)

    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=1)

    p.add_argument("--game-set", type=str, default="eval30", choices=["eval30", "tuning8"])
    p.add_argument("--games", type=str, default=None)

    # Match your PQN tuner defaults.
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--momentum", type=float, default=0.95)

    # Used only if missing from summary.
    p.add_argument("--q-lambda", type=float, default=0.65)
    p.add_argument("--exploration-fraction", type=float, default=0.10)

    # Optional explicit override. If omitted, child script default is used.
    p.add_argument("--end-e", type=float, default=None)

    p.add_argument("--wandb-project-name", type=str, default="cleanrl-atari-best-hparam-eval")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tag", type=str, default="pqn-best-hparam-eval")

    p.add_argument("--logs-root", type=str, default="eval_logs")
    p.add_argument("--summary-csv", type=str, default=None)

    p.add_argument("--cpus-per-run", type=int, default=8)

    p.add_argument("--track", action="store_true", default=True)
    p.add_argument("--no-track", action="store_false", dest="track")

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
            num_minibatches=args.num_minibatches,
            update_epochs=args.update_epochs,
            momentum=args.momentum,
            default_q_lambda=args.q_lambda,
            default_exploration_fraction=args.exploration_fraction,
            end_e=args.end_e,
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
                        track=args.track,
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
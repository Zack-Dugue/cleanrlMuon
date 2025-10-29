# file: multi_env_tuner_atari_dp.py
"""
Multi-GPU Atari tuner with checkpoint/resume.

Features
--------
• CPU / 1×GPU: single-process, logs to console.
• N×GPU (N>=2): shard games across GPUs, one worker per GPU, each logs to its own file.
  - If any worker errors, print to stderr and abort all.
• W&B support via env + CleanRL flags.
• Tunes momentum (Tyro flag --momentum), plus lr/ent_coef/update_epochs.
• Saves best shard results and a best_overall.json.
• NEW: Checkpoint/resume using Optuna persistent storage (SQLite by default).
  - --resume_dir to continue a previous run directory.
  - Or pass --storage_url (e.g., sqlite:///tuning.db) to keep a stable DB across runs.

Requirements
------------
- optuna
- cleanrl_utils.tuner.Tuner (CleanRL)
- PyTorch (for CUDA count)
- (optional) psutil for CPU pinning on Linux
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import optuna
from cleanrl_utils.tuner import Tuner

# -------- Path to your CleanRL PPO script (Tyro-based) --------
CLEANRL_PPO = r"C:\Users\dugue\PycharmProjects\cleanrlMuon\cleanrl\ppo_atari.py"

# -------- Tuning sets (unchanged) --------
TARGET_SCORES_TUNE: Dict[str, Tuple[float, float]] = {
    "AlienNoFrameskip-v4": (227.75, 7127.7),
    "AmidarNoFrameskip-v4": (5.77, 1719.5),
    "AssaultNoFrameskip-v4": (222.39, 742.0),
    "AsterixNoFrameskip-v4": (210.0, 8503.3),
    "BoxingNoFrameskip-v4": (0.05, 12.1),
    "BreakoutNoFrameskip-v4": (1.72, 30.5),
    "CentipedeNoFrameskip-v4": (2090.87, 12017.0),
    "DemonAttackNoFrameskip-v4": (152.07, 1971.0),
    "EnduroNoFrameskip-v4": (0.0, 860.5),
    "FreewayNoFrameskip-v4": (0.01, 29.6),
    "FrostbiteNoFrameskip-v4": (65.2, 4334.7),
    "GopherNoFrameskip-v4": (257.6, 2412.5),
    "HeroNoFrameskip-v4": (1026.97, 30826.4),
    "JamesbondNoFrameskip-v4": (29.0, 302.8),
    "KangarooNoFrameskip-v4": (52.0, 3035.0),
    "KrullNoFrameskip-v4": (1598.05, 2665.5),
    "MsPacmanNoFrameskip-v4": (307.3, 6951.6),
    "PongNoFrameskip-v4": (-20.71, 14.6),
    "SeaquestNoFrameskip-v4": (68.4, 42054.7),
    "SpaceInvadersNoFrameskip-v4": (148.3, 1668.7),
}

TARGET_SCORES_EVAL: Dict[str, Tuple[float, float]] = {
    "AsteroidsNoFrameskip-v4": (719.1, 47388.7),
    "AtlantisNoFrameskip-v4": (12850.0, 29028.1),
    "BankHeistNoFrameskip-v4": (14.2, 753.1),
    "BattleZoneNoFrameskip-v4": (2360.0, 37187.5),
    "BeamRiderNoFrameskip-v4": (363.88, 16926.5),
    "BerzerkNoFrameskip-v4": (123.65, 2630.4),
    "BowlingNoFrameskip-v4": (23.11, 160.7),
    "ChopperCommandNoFrameskip-v4": (811.0, 7387.8),
    "CrazyClimberNoFrameskip-v4": (10780.5, 35829.4),
    "DefenderNoFrameskip-v4": (2874.5, 18688.9),
    "DoubleDunkNoFrameskip-v4": (-18.55, -16.4),
    "FishingDerbyNoFrameskip-v4": (-91.71, -38.7),
    "GravitarNoFrameskip-v4": (173.0, 3351.4),
    "IceHockeyNoFrameskip-v4": (-11.15, 0.9),
    "KungFuMasterNoFrameskip-v4": (258.5, 22736.3),
    "MontezumaRevengeNoFrameskip-v4": (0.0, 4753.3),
    "NameThisGameNoFrameskip-v4": (2292.35, 8049.0),
    "PhoenixNoFrameskip-v4": (761.4, 7242.6),
    "PitfallNoFrameskip-v4": (-229.44, 6463.7),
    "PrivateEyeNoFrameskip-v4": (24.94, 69571.3),
    "QbertNoFrameskip-v4": (163.88, 13455.0),
    "RiverraidNoFrameskip-v4": (1338.5, 17118.0),
    "RoadRunnerNoFrameskip-v4": (11.5, 7845.0),
    "RobotankNoFrameskip-v4": (2.16, 11.9),
    "SkiingNoFrameskip-v4": (-17098.09, -4336.9),
    "SolarisNoFrameskip-v4": (1236.3, 12326.7),
    "StarGunnerNoFrameskip-v4": (664.0, 10250.0),
    "SurroundNoFrameskip-v4": (-9.99, 6.53),
    "TennisNoFrameskip-v4": (-23.84, -8.3),
    "TimePilotNoFrameskip-v4": (3568.0, 5229.2),
    "TutankhamNoFrameskip-v4": (11.43, 167.6),
    "UpNDownNoFrameskip-v4": (533.4, 11693.2),
    "VentureNoFrameskip-v4": (0.0, 1187.5),
    "VideoPinballNoFrameskip-v4": (0.0, 17667.9),
    "WizardOfWorNoFrameskip-v4": (563.5, 4756.5),
    "YarsRevengeNoFrameskip-v4": (3092.91, 54576.9),
    "ZaxxonNoFrameskip-v4": (32.5, 9173.3),
}

# -----------------------------
# Search space (adds momentum)
# -----------------------------
def default_params_fn(trial: optuna.Trial) -> dict:
    return {
        # Tunables (Tyro kebab-case flags)
        "learning-rate": trial.suggest_float("lr", 3e-5, 3e-3, log=True),
        "ent-coef": trial.suggest_float("ent_coef", 0.0, 0.02),
        "update-epochs": trial.suggest_int("update_epochs", 2, 8),
        "momentum": trial.suggest_float("momentum", 0.7, 0.99),  # tunes --momentum

        # Fixed batch shape
        "num-envs": 32,
        "num-steps": 256,               # total batch: 8192
        "total-timesteps": 5_000_000,
    }

# -----------------------------
# Utilities
# -----------------------------
def shard_evenly(items: List[str], parts: int) -> List[List[str]]:
    if parts <= 1:
        return [items]
    k, r = divmod(len(items), parts)
    out, i = [], 0
    for p in range(parts):
        sz = k + (1 if p < r else 0)
        out.append(items[i:i+sz])
        i += sz
    return out

def detect_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0

def prepare_wandb_env(enable: bool, project: str, entity: str | None) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if enable:
        env["WANDB_PROJECT"] = project
        if entity:
            env["WANDB_ENTITY"] = entity
        env["WANDB_SILENT"] = "true"
        env["WANDB__SERVICE_WAIT"] = "300"
    else:
        env["WANDB_MODE"] = "disabled"
    return env

def set_cpu_affinity(cpu_ids: List[int] | None) -> None:
    if not cpu_ids:
        return
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cpu_ids))
            return
        import psutil  # type: ignore
        psutil.Process().cpu_affinity(cpu_ids)
    except Exception:
        pass

def shard_study_name(base: str, games: List[str]) -> str:
    # Deterministic ID based on the game list (order-insensitive)
    key = ",".join(sorted(games))
    hid = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{hid}"

def default_sqlite_url(run_dir: str) -> str:
    # Use an on-disk SQLite DB inside the run directory
    db_path = Path(run_dir) / "optuna.db"
    return f"sqlite:///{db_path.as_posix()}"

# -----------------------------
# Per-GPU worker
# -----------------------------
def worker_entry(
    gpu_idx: int | None,
    games: List[str],
    run_dir: str,
    use_eval_set: bool,
    num_trials: int,
    num_seeds: int,
    metric: str,
    metric_last_n_average_window: int,
    direction: str,
    aggregation_type: str,
    wandb_enable: bool,
    wandb_project: str,
    wandb_entity: str | None,
    cpu_ids: List[int] | None,
    storage_url: str,
    base_study_name: str,
    returncode_sentinel: mp.Value,
):
    """
    Runs one Tuner shard. On multi-GPU, stdout/stderr are redirected to a per-shard log file.
    On error: write to stderr and set returncode sentinel.
    """
    shard = f"gpu{gpu_idx}" if gpu_idx is not None else "solo"
    shard_dir = Path(run_dir) / shard
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_path = shard_dir / f"{shard}.log"

    # Multi-gpu mode?
    multi_mode = gpu_idx is not None and os.environ.get("MULTI_GPU_MODE", "0") == "1"

    # Env
    env = prepare_wandb_env(wandb_enable, wandb_project, wandb_entity)
    if gpu_idx is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    if cpu_ids:
        env.setdefault("OMP_NUM_THREADS", str(len(cpu_ids)))
        env.setdefault("MKL_NUM_THREADS", str(len(cpu_ids)))
    set_cpu_affinity(cpu_ids)

    # Targets for this shard
    full_dict = TARGET_SCORES_EVAL if use_eval_set else TARGET_SCORES_TUNE
    target_scores = {k: full_dict[k] for k in games}

    # Optuna bits
    sampler = optuna.samplers.TPESampler(seed=0)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3)
    shard_name = shard_study_name(base_study_name, games)

    tuner = Tuner(
        script=CLEANRL_PPO,
        metric=metric,
        metric_last_n_average_window=metric_last_n_average_window,
        direction=direction,
        aggregation_type=aggregation_type,
        target_scores=target_scores,
        pruner=pruner,
        sampler=sampler,
        storage=storage_url,            # <<< persistent storage
        study_name=shard_name,          # <<< load_if_exists=True inside Tuner
        params_fn=lambda trial: {
            **default_params_fn(trial),
            **({
                "wandb-project-name": wandb_project,
                **({"wandb-entity": wandb_entity} if wandb_entity else {})
            } if wandb_enable else {})
        },
    )

    try:
        if multi_mode:
            with open(log_path, "a", encoding="utf-8", errors="replace") as logf:
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = logf  # type: ignore
                sys.stderr = logf  # type: ignore
                try:
                    os.environ.update(env)
                    tuner.tune(num_trials=num_trials, num_seeds=num_seeds)
                finally:
                    sys.stdout.flush(); sys.stderr.flush()
                    sys.stdout, sys.stderr = old_out, old_err
        else:
            os.environ.update(env)
            tuner.tune(num_trials=num_trials, num_seeds=num_seeds)
    except Exception as e:
        msg = f"[{shard}] ERROR: {e}"
        if multi_mode:
            try:
                tail = ""
                if log_path.exists():
                    tail = "".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-80:])
                print(msg, file=sys.stderr, flush=True)
                if tail:
                    print(f"[{shard}] --- log tail ---", file=sys.stderr, flush=True)
                    print(tail, file=sys.stderr, flush=True)
            except Exception:
                print(msg, file=sys.stderr, flush=True)
        else:
            print(msg, file=sys.stderr, flush=True)
        with returncode_sentinel.get_lock():
            returncode_sentinel.value = 1
        return

    # Persist this shard’s best trial summary (best_* files are additive; OK across resumes)
    best = None
    try:
        study = getattr(tuner, "study", None)
        if study and getattr(study, "best_trial", None) is not None:
            bt = study.best_trial
            best = {"value": bt.value, "params": bt.params, "number": bt.number}
        for attr in ("best_result", "best_params", "best_value"):
            if best is None and hasattr(tuner, attr):
                br = getattr(tuner, attr)
                if isinstance(br, dict) and "params" in br and "value" in br:
                    best = br
    except Exception:
        pass

    out_path = Path(run_dir) / f"best_{shard}.json"
    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        json.dump(best or {"value": None, "params": {}}, f, indent=2, ensure_ascii=False)

# -----------------------------
# Orchestrator
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_eval_set", action="store_true",
                    help="Use the 37-game eval set instead of the 20-game tune set")
    ap.add_argument("--num_trials", type=int, default=10)
    ap.add_argument("--num_seeds", type=int, default=3)
    ap.add_argument("--study_name", type=str, default="ppo_atari_tune")
    ap.add_argument("--storage_url", type=str, default=None,
                    help='Optuna storage URL (e.g. "sqlite:///tuning.db"). If not set, a sqlite DB is created in the run dir.')
    ap.add_argument("--resume_dir", type=str, default=None,
                    help="Resume a previous run directory (expects an optuna.db or valid --storage_url).")
    ap.add_argument("--cpu_per_gpu", type=int, default=None,
                    help="Override CPU cores per GPU (Linux only, best-effort)")
    ap.add_argument("--wandb_enable", type=lambda s: s.lower() in {"1","true","yes"}, default=True)
    ap.add_argument("--wandb_project", type=str, default="cleanRL")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--results_dir", type=str, default="tuning_results")
    args = ap.parse_args()

    games_dict = TARGET_SCORES_EVAL if args.use_eval_set else TARGET_SCORES_TUNE
    game_keys = list(games_dict.keys())

    # Determine run_dir and storage
    if args.resume_dir:
        run_dir = Path(args.resume_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[tuner] Resuming in: {run_dir.resolve()}")
        storage_url = args.storage_url or default_sqlite_url(str(run_dir))
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.results_dir) / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        storage_url = args.storage_url or default_sqlite_url(str(run_dir))
        print(f"[tuner] Starting new run in: {run_dir.resolve()}")

    # Save a little meta file to make resuming easy
    meta_path = run_dir / "run_meta.json"
    meta = {"storage_url": storage_url, "study_name_base": args.study_name,
            "use_eval_set": args.use_eval_set, "games": game_keys}
    try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass

    ngpu = detect_gpus()
    multi = ngpu >= 2
    print(f"[tuner] GPUs detected: {ngpu} | multi-GPU mode: {multi}")
    os.makedirs(run_dir / "logs" if hasattr(run_dir, "__truediv__") else Path(run_dir, "logs"), exist_ok=True)

    if not multi:
        # Single GPU/CPU
        rc = mp.Value("i", 0)
        os.environ["MULTI_GPU_MODE"] = "0"
        worker_entry(
            gpu_idx=None,
            games=game_keys,
            run_dir=str(run_dir),
            use_eval_set=args.use_eval_set,
            num_trials=args.num_trials,
            num_seeds=args.num_seeds,
            metric="charts/episodic_return",
            metric_last_n_average_window=50,
            direction="maximize",
            aggregation_type="median",
            wandb_enable=args.wandb_enable,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            cpu_ids=None,
            storage_url=storage_url,
            base_study_name=args.study_name,
            returncode_sentinel=rc,
        )
        # Collate best
        best_overall = None
        for f in Path(run_dir).glob("best_*.json"):
            try:
                data = json.loads(Path(f).read_text(encoding="utf-8", errors="replace"))
                if data.get("value") is None:
                    continue
                if (best_overall is None) or (data["value"] > best_overall["value"]):
                    best_overall = {"value": data["value"], "params": data["params"], "source": f.name}
            except Exception:
                pass
        out_path = Path(run_dir) / "best_overall.json"
        with open(out_path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(best_overall or {"value": None, "params": {}}, f, indent=2, ensure_ascii=False)
        print(f"[tuner] Best overall saved to: {out_path.resolve()}")
        if best_overall:
            print(f"[tuner] Best score: {best_overall['value']:.4f} from {best_overall['source']}")
            print(f"[tuner] Best params: {best_overall['params']}")
        sys.exit(rc.value)

    # Multi-GPU mode
    os.environ["MULTI_GPU_MODE"] = "1"

    # Shard games across GPUs
    shards = shard_evenly(game_keys, ngpu)
    for i, sh in enumerate(shards):
        print(f"  - shard {i}: {len(sh)} games")

    # Optional CPU slices per GPU (Linux best-effort)
    cpu_slices: List[List[int] | None] = []
    try:
        import psutil  # type: ignore
        total = psutil.cpu_count(logical=True) or os.cpu_count() or 0
        if total and total > 0:
            per = args.cpu_per_gpu or max(1, total // ngpu)
            for i in range(ngpu):
                lo, hi = i * per, min((i + 1) * per, total)
                cpu_slices.append(list(range(lo, hi)) if hi > lo else None)
        else:
            cpu_slices = [None] * ngpu
    except Exception:
        cpu_slices = [None] * ngpu

    # Launch workers
    returncode_sentinel = mp.Value("i", 0)
    procs: List[mp.Process] = []
    for idx, shard_games in enumerate(shards):
        p = mp.Process(
            target=worker_entry,
            args=(
                idx,                           # gpu_idx
                shard_games,                   # games
                str(run_dir),
                args.use_eval_set,
                args.num_trials,
                args.num_seeds,
                "charts/episodic_return",
                50,
                "maximize",
                "median",
                args.wandb_enable,
                args.wandb_project,
                args.wandb_entity,
                cpu_slices[idx] if idx < len(cpu_slices) else None,
                storage_url,
                args.study_name,
                returncode_sentinel,
            ),
            daemon=False,
        )
        procs.append(p)
        p.start()

    # Monitor; if any worker signals error, kill all.
    try:
        while True:
            all_done = True
            any_fail = False
            for p in procs:
                p.join(timeout=0.5)
                if p.is_alive():
                    all_done = False
                else:
                    if p.exitcode and p.exitcode != 0:
                        any_fail = True
            if returncode_sentinel.value != 0:
                any_fail = True
            if any_fail:
                print("[tuner] Detected error in one or more GPU workers. Terminating others...", file=sys.stderr, flush=True)
                for p in procs:
                    if p.is_alive():
                        try:
                            if os.name == "nt":
                                p.terminate()
                            else:
                                os.kill(p.pid, signal.SIGTERM)
                        except Exception:
                            pass
                for p in procs:
                    try:
                        p.join(timeout=2.0)
                    except Exception:
                        pass
                sys.exit(1)
            if all_done:
                break
    finally:
        for p in procs:
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass

    # Collate best across shards
    best_overall = None
    for f in Path(run_dir).glob("best_*.json"):
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8", errors="replace"))
            if data.get("value") is None:
                continue
            if (best_overall is None) or (data["value"] > best_overall["value"]):
                best_overall = {"value": data["value"], "params": data["params"], "source": f.name}
        except Exception:
            pass

    out_path = Path(run_dir) / "best_overall.json"
    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        json.dump(best_overall or {"value": None, "params": {}}, f, indent=2, ensure_ascii=False)

    print(f"[tuner] Best overall saved to: {out_path.resolve()}")
    if best_overall:
        print(f"[tuner] Best score: {best_overall['value']:.4f} from {best_overall['source']}")
        print(f"[tuner] Best params: {best_overall['params']}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Windows-friendly
    main()

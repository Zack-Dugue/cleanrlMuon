# -*- coding: utf-8 -*-
import json, os, sys, subprocess, shlex, multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

import optuna
from optuna.trial import TrialState
from cleanrl_utils.tuner import Tuner

# ---- Path to your CleanRL PPO script (Gym v0.26 “NoFrameskip-v4” IDs) ----
CLEANRL_PPO = r"C:\Users\dugue\PycharmProjects\cleanrlMuon\cleanrl\ppo_atari.py"

# -----------------------------
# Tuning set (20 games)
# -----------------------------
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

# -----------------------------
# Evaluation set (37 games)
# -----------------------------
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
# Search space (tunes momentum too)
# -----------------------------
def default_params_fn(trial: optuna.Trial) -> dict:
    return {
        "learning-rate": trial.suggest_float("lr", 3e-5, 3e-3, log=True),
        "ent-coef": trial.suggest_float("ent_coef", 0.0, 0.02),
        "update-epochs": trial.suggest_int("update_epochs", 2, 8),
        "momentum": trial.suggest_float("momentum", 0.5, 0.99),  # used as SGD momentum or Adam β1
        "num-envs": 32,
        "num-steps": 256,            # total batch: 8192
        "total-timesteps": 5_000_000,
        # add "optimizer": "Adam"/"Muon"/"AdaMuon" here if you want to lock it
    }

# -----------------------------
# Logging helpers
# -----------------------------
def mkdir_p(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def run_cmd(cmd: List[str], env: dict, log_file: Path, timeout: int | None = None) -> tuple[int, str, str]:
    """Run a subprocess with UTF-8 decoding and line-buffered logging (Windows-safe)."""
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"\n=== RUN: {' '.join(shlex.quote(c) for c in cmd)} ===\n")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,                # line-buffered
            encoding="utf-8",
            errors="replace",         # NEVER crash on odd bytes (Windows code pages)
        )
        out_lines, err_lines = [], []
        # stream stdout
        if proc.stdout is not None:
            for line in proc.stdout:
                f.write(line); out_lines.append(line)
        # stream stderr
        if proc.stderr is not None:
            for line in proc.stderr:
                f.write(line); err_lines.append(line)
        ret = proc.wait(timeout=timeout)
        return ret, "".join(out_lines), "".join(err_lines)

# -----------------------------
# Preflight: fail-fast diagnostics
# -----------------------------
def preflight_envs(env_ids: List[str], env: dict, logs_dir: Path, ppo_script: str) -> None:
    """1) Try gym.make for each v4 env; 2) run a 16-step PPO dry run (no W&B)."""
    log_envs = mkdir_p(logs_dir) / "preflight_envs.log"

    # 1) Env creation smoke test (Gym v0.26 ALE stack)
    check_code = (
        "import sys,gym\n"
        "ids=sys.argv[1:]\n"
        "ok=True\n"
        "for i in ids:\n"
        "  try:\n"
        "    gym.make(i).close(); print('OK', i)\n"
        "  except Exception as e:\n"
        "    ok=False; print('FAIL', i, e)\n"
        "sys.exit(0 if ok else 2)\n"
    )
    cmd = [sys.executable, "-c", check_code, *env_ids]
    ret, _, _ = run_cmd(cmd, env, log_envs)
    if ret != 0:
        raise RuntimeError(
            "Preflight FAILED: cannot create one or more Atari environments.\n"
            "Check ROMs and gym/ale-py install. See: "
            f"{log_envs}"
        )

    # 2) PPO dry run (tiny) for the first env id; catches CLI mismatches fast
    dry_env = env_ids[0]
    dry_cmd = [
        sys.executable, ppo_script,
        "--env-id", dry_env,
        "--total-timesteps", "16",
        "--num-envs", "2",
        "--num-steps", "8",
        "--track", "False",
    ]
    ret, out, err = run_cmd(dry_cmd, env, logs_dir / "preflight_dryrun.log", timeout=300)
    if ret != 0:
        lines = (err or out).splitlines()[:20]
        raise RuntimeError(
            "Preflight FAILED: PPO script cannot start with tiny settings.\n"
            "Common causes: unrecognized CLI flags, missing deps, wrong env IDs.\n"
            "First error lines:\n  " + "\n  ".join(lines) +
            f"\nSee log: {logs_dir / 'preflight_dryrun.log'}"
        )

# -----------------------------
# W&B + threading env (UTF-8 enforced)
# -----------------------------
def prepare_env_for_worker(
    enable_wandb: bool,
    project: str,
    entity: str | None,
    cpu_ids: List[int] | None
) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"         # real-time logs on SLURM/Windows
    # ---- FORCE UTF-8 EVERYWHERE (Windows-safe) ----
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("LC_ALL", "C.UTF-8")
    env.setdefault("LANG", "C.UTF-8")

    if enable_wandb:
        env["WANDB_PROJECT"] = project
        if entity:
            env["WANDB_ENTITY"] = entity
        env["WANDB_SILENT"] = "true"
        env["WANDB__SERVICE_WAIT"] = "300"
    else:
        env["WANDB_MODE"] = "disabled"

    # Optional CPU affinity (Linux best-effort)
    if cpu_ids:
        try:
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, set(cpu_ids))
        except Exception:
            pass
        env.setdefault("OMP_NUM_THREADS", str(len(cpu_ids)))
        env.setdefault("MKL_NUM_THREADS", str(len(cpu_ids)))
    return env

# -----------------------------
# Worker
# -----------------------------
def _worker(
    gpu_idx: int | None,
    shard_envs: List[str],
    study_name: str,
    storage_url: str | None,
    num_trials: int,
    num_seeds: int,
    metric: str,
    metric_last_n_average_window: int,
    direction: str,
    aggregation_type: str,
    params_fn,
    wandb_enable: bool,
    wandb_project: str,
    wandb_entity: str | None,
    cpu_ids: List[int] | None,
    results_dir: str,
):
    logs_dir = mkdir_p(Path(results_dir) / f"logs_worker_{'gpu'+str(gpu_idx) if gpu_idx is not None else 'cpu'}")
    worker_log = logs_dir / "worker.log"

    # Optuna verbosity
    optuna.logging.set_verbosity(optuna.logging.INFO)

    env = prepare_env_for_worker(wandb_enable, wandb_project, wandb_entity, cpu_ids)
    if gpu_idx is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    # ---- Preflight (fail fast with actionable errors) ----
    try:
        preflight_envs(shard_envs, env, logs_dir, CLEANRL_PPO)
    except Exception as e:
        msg = f"[preflight] {e}"
        worker_log.write_text(msg, encoding="utf-8")
        print(msg, flush=True)
        # Write an empty best shard file so the parent can proceed
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(results_dir)/f"best_{'gpu'+str(gpu_idx) if gpu_idx is not None else 'cpu'}.json","w",encoding="utf-8") as f:
            json.dump({"value": None, "params": {}, "error": str(e)}, f, indent=2)
        return

    # ---- Tuner setup ----
    sampler = optuna.samplers.TPESampler(seed=0)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3)
    target_scores = {k: TARGET_SCORES_TUNE[k] for k in shard_envs}

    tuner = Tuner(
        script=CLEANRL_PPO,
        metric=metric,
        metric_last_n_average_window=metric_last_n_average_window,
        direction=direction,
        aggregation_type=aggregation_type,
        target_scores=target_scores,
        pruner=pruner,
        sampler=sampler,
        # If your Tuner supports shared study across workers, uncomment:
        # storage=storage_url,
        # study_name=f"{study_name}_gpu{gpu_idx if gpu_idx is not None else 'cpu'}",
        params_fn=params_fn,
    )

    # Pass W&B flags into child CleanRL runs
    def params_with_tracking(trial: optuna.Trial) -> dict:
        p = params_fn(trial)
        if wandb_enable:
            p["track"] = True
            p["wandb-project-name"] = wandb_project
            if wandb_entity:
                p["wandb-entity"] = wandb_entity
        return p

    tuner.params_fn = params_with_tracking  # type: ignore

    # ---- Run tuning ----
    try:
        study = tuner.tune(num_trials=num_trials, num_seeds=num_seeds)
    except Exception as e:
        worker_log.write_text(f"[tuner] crashed: {e}", encoding="utf-8")
        print(f"[tuner] crashed: {e}", flush=True)
        study = None

    # ---- Persist best trial from this shard ----
    best = None
    try:
        if study is not None and hasattr(study, "trials"):
            good = [t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]
            if good:
                bt = max(good, key=lambda t: t.value if t.value is not None else float("-inf"))
                best = {"value": bt.value, "params": bt.params, "number": bt.number}
    except Exception:
        pass

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    shard = f"gpu{gpu_idx}" if gpu_idx is not None else "cpu"
    out_path = Path(results_dir) / f"best_{shard}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best or {"value": None, "params": {}}, f, indent=2)

# -----------------------------
# Utilities
# -----------------------------
def shard_evenly(items: List[str], parts: int) -> List[List[str]]:
    if parts <= 1: return [items]
    k, r = divmod(len(items), parts)
    out, s = [], 0
    for i in range(parts):
        size = k + (1 if i < r else 0)
        out.append(items[s:s+size]); s += size
    return out

def detect_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0

# -----------------------------
# Main
# -----------------------------
def main(
    use_eval_set: bool = False,
    num_trials: int = 10,
    num_seeds: int = 3,
    study_name: str = "ppo_atari_tune",
    storage_url: str | None = None,
    cpu_per_gpu: int | None = None,
    wandb_enable: bool = True,
    wandb_project: str = "cleanRL",
    wandb_entity: str | None = None,
    results_dir: str = "tuning_results",
):
    games_dict = (TARGET_SCORES_EVAL if use_eval_set else TARGET_SCORES_TUNE)
    game_keys = list(games_dict.keys())

    ngpu = detect_gpus()
    shards = shard_evenly(game_keys, ngpu) if ngpu >= 2 else [game_keys]

    # Optional CPU slices per shard (Linux best-effort)
    cpu_slices: List[List[int] | None] = [None] * len(shards)
    if ngpu >= 2:
        try:
            import psutil  # type: ignore
            total = psutil.cpu_count(logical=True) or os.cpu_count() or 0
            if total and total > 0:
                per = cpu_per_gpu or max(1, total // ngpu)
                for i in range(ngpu):
                    lo, hi = i * per, min((i + 1) * per, total)
                    cpu_slices[i] = list(range(lo, hi)) if hi > lo else None
        except Exception:
            pass

    metric = "charts/episodic_return"
    metric_window = 50
    direction = "maximize"
    aggregation_type = "median"

    procs: List[mp.Process] = []
    for idx, shard in enumerate(shards):
        gpu_idx = (idx if ngpu >= 2 else None)
        p = mp.Process(
            target=_worker,
            args=(gpu_idx, shard, study_name, storage_url, num_trials, num_seeds,
                  metric, metric_window, direction, aggregation_type,
                  default_params_fn, wandb_enable, wandb_project, wandb_entity,
                  cpu_slices[idx], results_dir),
            daemon=False,
        )
        p.start(); procs.append(p)
    for p in procs: p.join()

    # Collate best overall
    best_overall = None
    for f in Path(results_dir).glob("best_*.json"):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            if d.get("value") is None: continue
            if (best_overall is None) or (d["value"] > best_overall["value"]):
                d["source"] = f.name; best_overall = d
        except Exception:
            pass
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out = Path(results_dir) / "best_overall.json"
    out.write_text(json.dumps(best_overall or {"value": None, "params": {}}, indent=2), encoding="utf-8")
    print(f"[tuner] Best overall saved to: {out.resolve()}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_eval_set", action="store_true")
    ap.add_argument("--num_trials", type=int, default=10)
    ap.add_argument("--num_seeds", type=int, default=3)
    ap.add_argument("--study_name", type=str, default="ppo_atari_tune")
    ap.add_argument("--storage_url", type=str, default=None)
    ap.add_argument("--cpu_per_gpu", type=int, default=None)
    ap.add_argument("--wandb_enable", type=lambda s: s.lower() in {"1","true","yes"}, default=True)
    ap.add_argument("--wandb_project", type=str, default="cleanRL")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--results_dir", type=str, default="tuning_results")
    args = ap.parse_args()
    main(
        use_eval_set=args.use_eval_set,
        num_trials=args.num_trials,
        num_seeds=args.num_seeds,
        study_name=args.study_name,
        storage_url=args.storage_url,
        cpu_per_gpu=args.cpu_per_gpu,
        wandb_enable=args.wandb_enable,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        results_dir=args.results_dir,
    )

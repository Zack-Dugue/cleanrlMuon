"""
multi_gpu_tuner.py — DEBUG LOGGING EDITION

A spawn-safe, multi-GPU Optuna tuner that runs a CleanRL single-file script
across multiple environments and seeds, with extensive logging for headless servers.

Logging:
- Logs to stdout and to files in tuner_logs/{study_name}/
- Main controller: tuner.log
- One file per GPU worker: gpu{ID}-worker.log
- Set LOG_LEVEL=DEBUG (default) or INFO/WARNING/ERROR to adjust verbosity.

See the CLI at the bottom for a minimal ad-hoc run.
"""

from __future__ import annotations

import os
import sys
import time
import math
import glob
import runpy
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
from tensorboard.backend.event_processing import event_accumulator

try:
    import wandb
except Exception:
    wandb = None  # optional

import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import get_context

# ───────────────────────── Logging setup ─────────────────────────

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _setup_logger(name: str, log_file: str, level: int, to_stdout: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # don’t double log

    # Avoid duplicate handlers if re-invoked
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(processName)s[%(process)d] | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (rotating)
    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # Stdout handler
    if to_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(level)
        logger.addHandler(sh)

    return logger

def _get_log_level_from_env(default="DEBUG") -> int:
    return {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }.get(os.getenv("LOG_LEVEL", default).upper(), logging.DEBUG)

# Globals updated later
MAIN_LOGGER: Optional[logging.Logger] = None
WORKER_LOGGER: Optional[logging.Logger] = None

# ───────────────────────── Helper utilities ─────────────────────────

def _round_robin_split(items: List[str], n: int) -> List[List[str]]:
    buckets = [[] for _ in range(n)]
    for i, it in enumerate(items):
        buckets[i % n].append(it)
    return buckets

def _latest_runs_subdir(after_ts: float, base_dir: str = "runs", grace_s: float = 0.0) -> Optional[str]:
    try:
        candidates = []
        for d in glob.glob(os.path.join(base_dir, "*")):
            try:
                mtime = os.path.getmtime(d)
                if mtime >= (after_ts - grace_s) and os.path.isdir(d):
                    candidates.append((mtime, d))
            except Exception:
                pass
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return os.path.basename(candidates[0][1])
    except Exception:
        return None

# Per-worker config set by initializer
_WORKER_GPU = None
_WORKER_THREADS = None

def _init_worker(gpu_id: int, threads: int, study_name: str, logs_root: str, level: int):
    """
    Worker initializer: pin to GPU, set thread envs, and configure per-worker logger.
    """
    global _WORKER_GPU, _WORKER_THREADS, WORKER_LOGGER
    _WORKER_GPU = gpu_id
    _WORKER_THREADS = threads

    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_MAX_THREADS": str(threads),
        "BLIS_NUM_THREADS": str(threads),
    })

    worker_log_dir = os.path.join(logs_root, study_name)
    _ensure_dir(worker_log_dir)
    log_file = os.path.join(worker_log_dir, f"gpu{gpu_id}-worker.log")
    WORKER_LOGGER = _setup_logger(name=f"worker.gpu{gpu_id}", log_file=log_file, level=level, to_stdout=False)

    WORKER_LOGGER.info(f"Worker initialized on GPU {gpu_id}, threads={threads}")
    WORKER_LOGGER.debug(f"ENV: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    WORKER_LOGGER.debug(f"ENV threads = OMP:{os.environ.get('OMP_NUM_THREADS')} MKL:{os.environ.get('MKL_NUM_THREADS')} "
                        f"OPENBLAS:{os.environ.get('OPENBLAS_NUM_THREADS')} NUMEXPR:{os.environ.get('NUMEXPR_MAX_THREADS')} "
                        f"BLIS:{os.environ.get('BLIS_NUM_THREADS')}")

def _read_tb_metric(logdir: str, metric: str, last_n: int, retries: int = 5, sleep_s: float = 0.5, logger: Optional[logging.Logger]=None) -> float:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            if logger: logger.debug(f"[TB] Attempt {attempt}/{retries}: loading {logdir}, metric='{metric}'")
            ea = event_accumulator.EventAccumulator(logdir)
            ea.Reload()
            scalars = ea.Scalars(metric)
            if not scalars:
                raise FileNotFoundError(f"Metric '{metric}' not found in {logdir}")
            if last_n > 0:
                vals = [e.value for e in scalars[-last_n:]]
            else:
                vals = [e.value for e in scalars]
            avg = float(np.average(vals))
            if logger: logger.debug(f"[TB] Found {len(vals)} values, avg(last {last_n if last_n>0 else 'all'})={avg:.6f}")
            return avg
        except Exception as e:
            last_err = e
            if logger: logger.warning(f"[TB] Read failed (attempt {attempt}): {e}")
            time.sleep(sleep_s)
    if logger:
        logger.error(f"[TB] Failed to read metric after {retries} attempts. Last error:\n{traceback.format_exc()}")
    raise last_err if last_err else RuntimeError(f"Failed to read TB metric {metric} from {logdir}")

def _run_one_env(
    env_id: str,
    script: str,
    metric: str,
    algo_argv: List[str],
    seed: int,
    tb_window: int,
    study_name: str,
    logs_root: str,
    level: int,
) -> Tuple[str, float, str]:
    """
    Run one CleanRL script for a single environment and seed in this worker.
    Returns: (env_id, metric_value, run_name)
    """
    logger = WORKER_LOGGER
    if logger is None:
        # Fallback logger (shouldn't happen if _init_worker ran)
        tmp_dir = os.path.join(logs_root, study_name)
        _ensure_dir(tmp_dir)
        logger = _setup_logger(f"worker.fallback", os.path.join(tmp_dir, "worker-fallback.log"), level, to_stdout=False)

    argv = algo_argv + [f"--env-id={env_id}", f"--seed={seed}"]
    sys.argv = argv

    logger.info(f"[RUN] Starting env='{env_id}' seed={seed} on GPU={_WORKER_GPU}")
    logger.debug(f"[RUN] sys.argv = {sys.argv}")

    start_ts = time.time()
    try:
        g = runpy.run_path(path_name=script, run_name="__main__")
    except SystemExit as e:
        # Some scripts call sys.exit(); still attempt to proceed
        logger.warning(f"[RUN] Script raised SystemExit({e.code}); continuing to locate logs.")
        g = {}
    except Exception:
        logger.error(f"[RUN] Exception while running script:\n{traceback.format_exc()}")
        raise

    run_name = g.get("run_name", None)
    if run_name:
        logger.debug(f"[RUN] Detected run_name from globals: {run_name}")
    else:
        rn = _latest_runs_subdir(after_ts=start_ts, base_dir="runs", grace_s=1.0)
        if rn is None:
            logger.error("[RUN] Unable to determine run_name; ensure your script sets `run_name` or writes to 'runs/'.")
            raise RuntimeError("run_name not found and fallback failed.")
        run_name = rn
        logger.debug(f"[RUN] Fallback run_name via filesystem: {run_name}")

    logdir = os.path.join("runs", run_name)
    logger.info(f"[TB] Reading metric '{metric}' from {logdir} (window={tb_window})")
    metric_val = _read_tb_metric(logdir=logdir, metric=metric, last_n=tb_window, logger=logger)

    dur = time.time() - start_ts
    logger.info(f"[DONE] env='{env_id}' seed={seed} metric={metric_val:.6f} (elapsed {dur:.2f}s)")
    return env_id, metric_val, run_name

# ───────────────────────── Tuner class ─────────────────────────

class MultiGPUTuner:
    def __init__(
        self,
        script: str,
        metric: str,
        gpus: List[int],
        target_scores: Dict[str, Optional[List[float]]],
        params_fn: Callable[[optuna.Trial], Dict],
        direction: str = "maximize",
        aggregation_type: str = "average",
        metric_last_n_average_window: int = 50,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        storage: str = "sqlite:///cleanrl_hpopt.db",
        study_name: str = "",
        wandb_kwargs: Dict[str, any] = {},
        logs_root: str = "tuner_logs",
    ) -> None:
        if not gpus:
            raise ValueError("No GPUs specified.")
        if len(target_scores) > 1 and any(v is None for v in target_scores.values()):
            raise ValueError("Multiple environments provided — specify [min, max] normalization range for each.")

        self.script = script
        self.metric = metric
        self.target_scores = target_scores
        self.gpus = list(gpus)
        self.n_gpus = len(self.gpus)
        self.params_fn = params_fn
        self.direction = direction
        self.logs_root = logs_root

        agg = aggregation_type.lower()
        if agg == "average":
            self.aggregation_fn = np.average
        elif agg == "median":
            self.aggregation_fn = np.median
        elif agg == "max":
            self.aggregation_fn = np.max
        elif agg == "min":
            self.aggregation_fn = np.min
        else:
            raise ValueError(f"Unknown aggregation type '{aggregation_type}'")
        self.aggregation_type = agg

        self.metric_last_n_average_window = int(metric_last_n_average_window)
        self.pruner = pruner
        self.sampler = sampler
        self.storage = storage
        self.study_name = study_name or f"tuner_{int(time.time())}"
        self.wandb_kwargs = wandb_kwargs

        # CPU sanity
        n_cpus = os.cpu_count() or 1
        self.n_cpus = max(1, n_cpus - 1)
        if self.n_cpus < self.n_gpus:
            raise ValueError(f"Need at least one CPU core per GPU. CPUs: {self.n_cpus}, GPUs: {self.n_gpus}")

        # Main logger
        level = _get_log_level_from_env()
        study_log_dir = os.path.join(self.logs_root, self.study_name)
        _ensure_dir(study_log_dir)
        global MAIN_LOGGER
        MAIN_LOGGER = _setup_logger("tuner", os.path.join(study_log_dir, "tuner.log"), level, to_stdout=True)

        # Environment snapshot
        MAIN_LOGGER.info(f"Initialized MultiGPUTuner study='{self.study_name}' dir='{study_log_dir}'")
        MAIN_LOGGER.info(f"GPUs={self.gpus} CPUs={self.n_cpus} threads_per_gpu≈{max(1, self.n_cpus // self.n_gpus)}")
        MAIN_LOGGER.info(f"Script='{self.script}', Metric='{self.metric}', Agg='{self.aggregation_type}'")
        MAIN_LOGGER.info(f"Envs={list(self.target_scores.keys())}")
        MAIN_LOGGER.debug(f"Sys: python={sys.version}")
        try:
            import torch  # type: ignore
            MAIN_LOGGER.debug(f"Torch: version={torch.__version__} cuda_available={torch.cuda.is_available()}")
        except Exception:
            MAIN_LOGGER.debug("Torch not importable (ok).")

    def _normalize(self, env_id: str, value: float) -> float:
        lo_hi = self.target_scores.get(env_id)
        if lo_hi is None:
            return float(value)
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
        if hi == lo:
            return 0.0
        return float((value - lo) / (hi - lo))

    def tune(self, num_trials: int, num_seeds: int):
        env_ids = list(self.target_scores.keys())
        level = _get_log_level_from_env()

        def objective(trial: optuna.Trial):
            start_trial = time.time()

            # Params from trial
            params = self.params_fn(trial)
            algo_argv = [f"--{k}={v}" for k, v in params.items()]
            MAIN_LOGGER.info(f"[TRIAL {trial.number}] Params: {params}")

            # Optional W&B in parent only
            run = None
            if self.wandb_kwargs and wandb is not None and len(self.wandb_kwargs) > 0:
                MAIN_LOGGER.info(f"[TRIAL {trial.number}] Initializing Weights & Biases run")
                run = wandb.init(
                    **self.wandb_kwargs,
                    config=params,
                    name=f"{self.study_name}_{trial.number}",
                    group=self.study_name,
                    save_code=True,
                    reinit=True,
                )

            threads = max(1, self.n_cpus // self.n_gpus)
            mp_ctx = get_context("spawn")
            MAIN_LOGGER.debug(f"[TRIAL {trial.number}] Using spawn context; threads/gpu={threads}")

            per_seed_aggregates: List[float] = []

            for seed in range(num_seeds):
                MAIN_LOGGER.info(f"[TRIAL {trial.number}] Seed {seed} starting")
                shards = _round_robin_split(env_ids, self.n_gpus)
                for i, shard in enumerate(shards):
                    MAIN_LOGGER.debug(f"[TRIAL {trial.number}] Seed {seed} GPU {self.gpus[i]} shard: {shard}")

                pools = []
                results: List[Tuple[str, float, str]] = []

                # Create one single-process pool per GPU
                try:
                    for gpu_id in self.gpus:
                        p = mp_ctx.Pool(
                            processes=1,
                            initializer=_init_worker,
                            initargs=(gpu_id, threads, self.study_name, self.logs_root, level),
                        )
                        pools.append(p)
                        MAIN_LOGGER.debug(f"[TRIAL {trial.number}] Spawned worker pool for GPU {gpu_id}")

                    # Dispatch envs
                    for pool, shard, gpu_id in zip(pools, shards, self.gpus):
                        if not shard:
                            MAIN_LOGGER.debug(f"[TRIAL {trial.number}] GPU {gpu_id} shard empty; skipping")
                            continue
                        jobs = [(env, self.script, self.metric, algo_argv, seed,
                                 self.metric_last_n_average_window, self.study_name, self.logs_root, level)
                                for env in shard]
                        MAIN_LOGGER.info(f"[TRIAL {trial.number}] GPU {gpu_id} executing {len(jobs)} job(s)")
                        try:
                            res = pool.starmap(_run_one_env, jobs)
                            results.extend(res)
                        except Exception:
                            MAIN_LOGGER.error(f"[TRIAL {trial.number}] Exception during pool.starmap on GPU {gpu_id}:\n{traceback.format_exc()}")
                            raise
                finally:
                    # Cleanup
                    for p in pools:
                        try:
                            p.close()
                        except Exception:
                            pass
                    for p in pools:
                        try:
                            p.join()
                        except Exception:
                            pass

                # Aggregate per seed
                if not results:
                    MAIN_LOGGER.error(f"[TRIAL {trial.number}] No results collected for seed {seed}")
                    raise RuntimeError("No results collected for this seed; check script/logging paths.")

                per_env_norm = []
                for (env_id, raw_val, run_name) in results:
                    norm = self._normalize(env_id, raw_val)
                    per_env_norm.append(norm)
                    MAIN_LOGGER.info(f"[TRIAL {trial.number}] Seed {seed} {env_id}: raw={raw_val:.6f} "
                                     f"norm={norm:.6f} run_name={run_name}")
                    if run is not None:
                        run.log({f"{env_id}_return": raw_val})

                aggregated = float(self.aggregation_fn(per_env_norm))
                per_seed_aggregates.append(aggregated)
                MAIN_LOGGER.info(f"[TRIAL {trial.number}] Seed {seed} aggregated ({self.aggregation_type}) = {aggregated:.6f}")

                trial.report(aggregated, step=seed)
                if run is not None:
                    run.log({"aggregated_normalized_score": aggregated, "seed": seed})

                if trial.should_prune():
                    MAIN_LOGGER.warning(f"[TRIAL {trial.number}] Pruned at seed {seed}")
                    if run is not None:
                        run.finish(quiet=True)
                    raise optuna.TrialPruned()

            final_score = float(np.average(per_seed_aggregates))
            MAIN_LOGGER.info(f"[TRIAL {trial.number}] FINAL score (avg over {num_seeds} seeds) = {final_score:.6f} "
                             f"in {time.time() - start_trial:.2f}s")
            if run is not None:
                run.log({"final_score": final_score})
                run.finish(quiet=True)
            return final_score

        # Build/Resume study
        level = _get_log_level_from_env()
        study_log_dir = os.path.join(self.logs_root, self.study_name)
        _ensure_dir(study_log_dir)
        MAIN_LOGGER.info("=" * 90)
        MAIN_LOGGER.info("Starting Optuna study.optimize(...)")
        MAIN_LOGGER.info("=" * 90)

        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            pruner=self.pruner,
            sampler=self.sampler,
            load_if_exists=True,
        )

        try:
            study.optimize(objective, n_trials=num_trials)
        except KeyboardInterrupt:
            MAIN_LOGGER.warning("KeyboardInterrupt received; stopping study gracefully.")
        finally:
            if len(study.trials) > 0:
                MAIN_LOGGER.info(f"Trials completed: {len(study.trials)}; Best value so far: {getattr(study.best_trial, 'value', None)}")
            else:
                MAIN_LOGGER.info("No trials completed.")

        MAIN_LOGGER.info(f"BEST TRIAL: value={study.best_trial.value} params={study.best_trial.params}")
        return study.best_trial

# ───────────────────────── CLI for ad-hoc runs ─────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, required=True, help="Path to CleanRL single-file script")
    parser.add_argument("--metric", type=str, required=True, help="TensorBoard scalar name to read")
    parser.add_argument("--gpus", type=str, required=True, help="Comma-separated CUDA device indices, e.g., 0,1")
    parser.add_argument("--envs", type=str, required=True, nargs="+",
                        help="Space-separated list of env ids; if you want normalization, see --norms")
    parser.add_argument("--norms", type=str, default="",
                        help="Optional normalization ranges 'env:min:max,env2:min:max' (omit to use raw metric)")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--logs-root", type=str, default="tuner_logs")
    args = parser.parse_args()

    # Minimal params_fn for quick testing; customize for real studies
    def _params_fn(_trial: optuna.Trial) -> Dict:
        # The tuner will inject --env-id and --seed.
        return {
            "total-timesteps": int(1e6),
            "learning-rate": _trial.suggest_float("learning-rate", 1e-4, 3e-3, log=True),
        }

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    target_scores: Dict[str, Optional[List[float]]] = {e: None for e in args.envs}
    if args.norms:
        # format: env:min:max,env2:min:max
        for chunk in args.norms.split(","):
            if not chunk:
                continue
            env, lo, hi = chunk.split(":")
            target_scores[env] = [float(lo), float(hi)]

    tuner = MultiGPUTuner(
        script=args.script,
        metric=args.metric,
        gpus=gpu_list,
        target_scores=target_scores,
        params_fn=_params_fn,
        direction="maximize",
        aggregation_type="average",
        metric_last_n_average_window=50,
        sampler=optuna.samplers.TPESampler(seed=123),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        storage="sqlite:///cleanrl_hpopt.db",
        study_name="cli_multi_gpu_tuner",
        wandb_kwargs={},  # fill to enable W&B
        logs_root=args.logs_root,
    )

    best = tuner.tune(num_trials=args.trials, num_seeds=args.seeds)
    print("Best:", best.value, best.params)

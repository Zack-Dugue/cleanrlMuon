"""
multi_gpu_tuner.py — DEBUG LOGGING EDITION (Fail-Fast Enhanced + .trk progress)

Adds a third log file type ".trk" that appends a one-line progress record whenever
a single env run completes (i.e., one (env_id, seed) finishes on some GPU).

Example .trk line:
  2025-11-02 12:34:56 | done 7/120 | elapsed 00:03:41 | trial 3 seed 0 env Pong-v5

Use:
  watch -n 1 tail -n 20 tuner_logs/<study_name>/<study_name>.trk
"""

from __future__ import annotations
import os, sys, time, math, glob, runpy, traceback, logging, threading, socket
from logging.handlers import RotatingFileHandler
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np, optuna
from tensorboard.backend.event_processing import event_accumulator
from multiprocessing import get_context

try:
    import wandb
except Exception:
    wandb = None

# ───────────────────────── Logging setup ─────────────────────────

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _setup_logger(name: str, log_file: str, level: int, to_stdout: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(processName)s[%(process)d] | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)
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

MAIN_LOGGER: Optional[logging.Logger] = None
WORKER_LOGGER: Optional[logging.Logger] = None

# ───────────────────────── Helper utilities ─────────────────────────

def _round_robin_split(items: List[str], n: int) -> List[List[str]]:
    buckets = [[] for _ in range(n)]
    for i, it in enumerate(items):
        buckets[i % n].append(it)
    return buckets

def _latest_runs_subdir(after_ts: float, base_dir="runs", grace_s: float = 0.0) -> Optional[str]:
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

def _hhmmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

_WORKER_GPU = None
_WORKER_THREADS = None

def _init_worker(gpu_id: int, threads: int, study_name: str, logs_root: str, level: int):
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
    WORKER_LOGGER = _setup_logger(f"worker.gpu{gpu_id}", log_file, level, to_stdout=False)
    WORKER_LOGGER.info(f"Worker initialized on GPU {gpu_id}, threads={threads}")

def _read_tb_metric(logdir: str, metric: str, last_n: int,
                    retries: int = 5, sleep_s: float = 0.5,
                    logger: Optional[logging.Logger] = None) -> float:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            if logger: logger.debug(f"[TB] Attempt {attempt}/{retries}: {logdir}")
            ea = event_accumulator.EventAccumulator(logdir)
            ea.Reload()
            scalars = ea.Scalars(metric)
            if not scalars:
                raise FileNotFoundError(f"Metric '{metric}' not found")
            vals = [e.value for e in scalars[-last_n:]] if last_n > 0 else [e.value for e in scalars]
            return float(np.average(vals))
        except Exception as e:
            last_err = e
            if logger: logger.warning(f"[TB] Read failed (attempt {attempt}): {e}")
            time.sleep(sleep_s)
    if logger: logger.error(f"[TB] Failed after {retries} attempts\n{traceback.format_exc()}")
    raise last_err if last_err else RuntimeError(f"TB metric {metric} read failed")

def _run_one_env(env_id: str, script: str, metric: str, algo_argv: List[str], seed: int,
                 tb_window: int, study_name: str, logs_root: str, level: int) -> Tuple[str, float, str]:
    logger = WORKER_LOGGER or _setup_logger(
        "worker.fallback", os.path.join(logs_root, study_name, "worker-fallback.log"), level, to_stdout=False)
    argv = algo_argv + [f"--env-id={env_id}", f"--seed={seed}"]
    sys.argv = argv
    logger.info(f"[RUN] env='{env_id}' seed={seed} GPU={_WORKER_GPU}")
    start_ts = time.time()
    try:
        g = runpy.run_path(path_name=script, run_name="__main__")
    except SystemExit as e:
        logger.warning(f"[RUN] SystemExit({e.code}) caught")
        g = {}
    except Exception:
        logger.error(f"[RUN] Exception:\n{traceback.format_exc()}")
        raise
    run_name = g.get("run_name") or _latest_runs_subdir(after_ts=start_ts, base_dir="runs", grace_s=1.0)
    if not run_name:
        logger.error("[RUN] run_name not found")
        raise RuntimeError("run_name not found")
    logdir = os.path.join("runs", run_name)
    metric_val = _read_tb_metric(logdir, metric, tb_window, logger=logger)
    logger.info(f"[DONE] {env_id}={metric_val:.6f} ({time.time()-start_ts:.2f}s)")
    return env_id, metric_val, run_name

# ───────────────────────── Tuner ─────────────────────────

class MultiGPUTuner:
    def __init__(self, script: str, metric: str, gpus: List[int],
                 target_scores: Dict[str, Optional[List[float]]],
                 params_fn: Callable[[optuna.Trial], Dict],
                 direction="maximize", aggregation_type="average",
                 metric_last_n_average_window=50,
                 sampler=None, pruner=None,
                 storage="sqlite:///cleanrl_hpopt.db",
                 study_name="", wandb_kwargs=None, logs_root="tuner_logs") -> None:

        wandb_kwargs = wandb_kwargs or {}
        self.script, self.metric = script, metric
        self.target_scores, self.gpus = target_scores, list(gpus)
        self.n_gpus = len(self.gpus)
        self.params_fn, self.direction = params_fn, direction
        self.logs_root = logs_root

        agg = aggregation_type.lower()
        self.aggregation_fn = getattr(np, agg if agg in ["average", "median", "max", "min"] else "average")
        self.aggregation_type = agg
        self.metric_last_n_average_window = int(metric_last_n_average_window)
        self.pruner, self.sampler = pruner, sampler
        self.storage = storage
        self.study_name = study_name or f"tuner_{int(time.time())}"
        self.wandb_kwargs = wandb_kwargs

        # CPU sanity
        n_cpus = os.cpu_count() or 1
        self.n_cpus = max(1, n_cpus - 1)
        if self.n_cpus < self.n_gpus:
            raise ValueError("Need >= 1 CPU per GPU")

        # Loggers
        level = _get_log_level_from_env()
        study_log_dir = os.path.join(self.logs_root, self.study_name)
        _ensure_dir(study_log_dir)
        global MAIN_LOGGER
        MAIN_LOGGER = _setup_logger("tuner", os.path.join(study_log_dir, "tuner.log"), level, to_stdout=True)
        MAIN_LOGGER.info(f"Init study={self.study_name} GPUs={self.gpus} CPUs={self.n_cpus}")

        # ── Progress tracker (.trk)
        self._trk_path = os.path.join(study_log_dir, f"{self.study_name}.trk")
        self._progress_lock = threading.Lock()
        self._runs_total = None        # set in tune()
        self._runs_completed = 0
        self._tune_start_ts = None

        # Include node name for multi-node log separation (optional)
        self.node_name = os.environ.get("SLURMD_NODENAME", socket.gethostname())
        with open(self._trk_path, "a", encoding="utf-8") as f:
            f.write(f"# study={self.study_name} node={self.node_name}\n")

    def _append_trk(self, msg: str):
        """Append a single line to the .trk progress file."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self._trk_path, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {msg}\n")

    def _normalize(self, env_id: str, value: float) -> float:
        lo_hi = self.target_scores.get(env_id)
        if not lo_hi: return float(value)
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
        return 0.0 if hi == lo else float((value - lo) / (hi - lo))

    def tune(self, num_trials: int, num_seeds: int):
        env_ids = list(self.target_scores.keys())
        level = _get_log_level_from_env()

        # Study-level totals for .trk
        self._runs_total = int(num_trials) * int(num_seeds) * len(env_ids)
        self._runs_completed = 0
        self._tune_start_ts = time.time()
        self._append_trk(f"start 0/{self._runs_total} | elapsed 00:00:00")

        def objective(trial: optuna.Trial):
            start_trial = time.time()
            params = self.params_fn(trial)
            algo_argv = [f"--{k}={v}" for k, v in params.items()]
            MAIN_LOGGER.info(f"[TRIAL {trial.number}] Params: {params}")

            run = None
            if self.wandb_kwargs and wandb:
                run = wandb.init(**self.wandb_kwargs, config=params,
                                 name=f"{self.study_name}_{trial.number}",
                                 group=self.study_name, save_code=True, reinit=True)
            threads = max(1, self.n_cpus // self.n_gpus)
            mp_ctx = get_context("spawn")
            per_seed_aggregates: List[float] = []

            for seed in range(num_seeds):
                MAIN_LOGGER.info(f"[TRIAL {trial.number}] Seed {seed} starting")
                shards = _round_robin_split(env_ids, self.n_gpus)
                pools, results = [], []

                # callback to record each job completion into .trk
                def _on_done(res_tuple):
                    # res_tuple: (env_id, raw_val, run_name)
                    with self._progress_lock:
                        self._runs_completed += 1
                        elapsed = _hhmmss(time.time() - self._tune_start_ts)
                        env = res_tuple[0] if isinstance(res_tuple, (list, tuple)) and res_tuple else "?"
                        self._append_trk(
                            f"done {self._runs_completed}/{self._runs_total} | elapsed {elapsed} | "
                            f"trial {trial.number} seed {seed} env {env}"
                        )

                def _on_err(exc):
                    # also reflect errors in .trk for visibility
                    with self._progress_lock:
                        elapsed = _hhmmss(time.time() - self._tune_start_ts)
                        self._append_trk(
                            f"error at {self._runs_completed}/{self._runs_total} | elapsed {elapsed} | "
                            f"trial {trial.number} seed {seed} | {repr(exc)}"
                        )
                    # Log to main logger and re-raise to fail-fast
                    MAIN_LOGGER.error(f"[TRIAL {trial.number}] Worker error:\n{traceback.format_exc()}")

                try:
                    # One single-process pool per GPU
                    for gpu_id in self.gpus:
                        p = mp_ctx.Pool(
                            processes=1,
                            initializer=_init_worker,
                            initargs=(gpu_id, threads, self.study_name, self.logs_root, level),
                        )
                        pools.append(p)

                    # Submit jobs individually so we can get per-completion callbacks
                    async_results = []
                    for pool, shard, gpu_id in zip(pools, shards, self.gpus):
                        if not shard:
                            continue
                        jobs = [(env, self.script, self.metric, algo_argv, seed,
                                 self.metric_last_n_average_window, self.study_name,
                                 self.logs_root, level) for env in shard]
                        MAIN_LOGGER.info(f"[TRIAL {trial.number}] GPU {gpu_id} → {len(jobs)} job(s)")
                        for job_args in jobs:
                            ar = pool.apply_async(
                                _run_one_env,
                                job_args,
                                callback=lambda res, _on_done=_on_done, _results=results: (_results.append(res), _on_done(res)),
                                error_callback=_on_err,
                            )
                            async_results.append(ar)

                    # Wait for all outstanding jobs in this seed to complete (or error)
                    for ar in async_results:
                        try:
                            res = ar.get()  # already appended in callback, but ensures exceptions propagate
                        except Exception:
                            # already logged and tracked; raise to fail the trial
                            raise

                finally:
                    for p in pools:
                        try: p.close()
                        except Exception: pass
                    for p in pools:
                        try: p.join()
                        except Exception: pass

                # Fail fast if nothing came back
                if not results:
                    MAIN_LOGGER.error(f"[TRIAL {trial.number}] No results for seed {seed} — aborting.")
                    raise RuntimeError("Worker failure; terminating study.")

                # Aggregate
                per_env_norm = []
                for (env_id, raw_val, run_name) in results:
                    norm = self._normalize(env_id, raw_val)
                    per_env_norm.append(norm)
                    MAIN_LOGGER.info(f"[TRIAL {trial.number}] Seed {seed} {env_id}: raw={raw_val:.6f} norm={norm:.6f}")
                    if run: run.log({f"{env_id}_return": raw_val})
                aggregated = float(self.aggregation_fn(per_env_norm))
                per_seed_aggregates.append(aggregated)
                MAIN_LOGGER.info(f"[TRIAL {trial.number}] Seed {seed} aggregate={aggregated:.6f}")
                trial.report(aggregated, step=seed)
                if run: run.log({"aggregated_normalized_score": aggregated, "seed": seed})
                if trial.should_prune():
                    MAIN_LOGGER.warning(f"[TRIAL {trial.number}] Pruned at seed {seed}")
                    if run: run.finish(quiet=True)
                    raise optuna.TrialPruned()

            final_score = float(np.average(per_seed_aggregates))
            MAIN_LOGGER.info(f"[TRIAL {trial.number}] FINAL score={final_score:.6f} "
                             f"({time.time()-start_trial:.2f}s)")
            if run:
                run.log({"final_score": final_score})
                run.finish(quiet=True)
            return final_score

        level = _get_log_level_from_env()
        study_log_dir = os.path.join(self.logs_root, self.study_name)
        _ensure_dir(study_log_dir)
        MAIN_LOGGER.info("="*90)
        MAIN_LOGGER.info("Starting Optuna study.optimize(...)")
        MAIN_LOGGER.info("="*90)

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
        except Exception:
            MAIN_LOGGER.error("Fatal error during tuning:\n" + traceback.format_exc())
            sys.exit(1)
        finally:
            if len(study.trials) > 0:
                MAIN_LOGGER.info(f"Trials completed: {len(study.trials)}; Best={getattr(study.best_trial, 'value', None)}")
            else:
                MAIN_LOGGER.info("No trials completed.")
        MAIN_LOGGER.info(f"BEST TRIAL: value={study.best_trial.value} params={study.best_trial.params}")
        return study.best_trial

# ───────────────────────── CLI ─────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--envs", required=True, nargs="+")
    parser.add_argument("--norms", default="")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--logs-root", default="tuner_logs")
    args = parser.parse_args()

    def _params_fn(_trial: optuna.Trial) -> Dict:
        return {
            "total-timesteps": int(1e6),
            "learning-rate": _trial.suggest_float("learning-rate", 1e-4, 3e-3, log=True),
        }

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    target_scores: Dict[str, Optional[List[float]]] = {e: None for e in args.envs}
    if args.norms:
        for chunk in args.norms.split(","):
            if not chunk: continue
            env, lo, hi = chunk.split(":")
            target_scores[env] = [float(lo), float(hi)]

    tuner = MultiGPUTuner(
        script=args.script, metric=args.metric,
        gpus=gpu_list, target_scores=target_scores,
        params_fn=_params_fn, direction="maximize",
        aggregation_type="average", metric_last_n_average_window=50,
        sampler=optuna.samplers.TPESampler(seed=123),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        storage="sqlite:///cleanrl_hpopt.db",
        study_name="cli_multi_gpu_tuner", wandb_kwargs={},
        logs_root=args.logs_root,
    )

    best = tuner.tune(num_trials=args.trials, num_seeds=args.seeds)
    print("Best:", best.value, best.params)

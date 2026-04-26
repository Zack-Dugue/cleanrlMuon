class MultiGPUTuner:
    def __init__(self, script: str, metric: str, gpus: List[int],
                 target_scores: Dict[str, Optional[List[float]]],
                 params_fn: Callable[[optuna.Trial], Dict],
                 direction="maximize", aggregation_type="average",
                 metric_last_n_average_window=50,
                 sampler=None, pruner=None,
                 storage="sqlite:///cleanrl_hpopt.db",
                 study_name="", wandb_kwargs=None, logs_root="tuner_logs",
                 wandb_tag: Optional[str] = None) -> None:

        wandb_kwargs = wandb_kwargs or {}
        self.script, self.metric = script, metric
        self.target_scores, self.gpus = target_scores, list(gpus)
        self.n_gpus = len(self.gpus)
        self.params_fn, self.direction = params_fn, direction
        self.logs_root = logs_root
        self.wandb_tag = wandb_tag

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
        self._runs_total = None
        self._runs_completed = 0
        self._tune_start_ts = None

        self.node_name = os.environ.get("SLURMD_NODENAME", socket.gethostname())
        with open(self._trk_path, "a", encoding="utf-8") as f:
            f.write(f"# study={self.study_name} node={self.node_name}\n")

    def _append_trk(self, msg: str):
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
                wandb_init_kwargs = dict(self.wandb_kwargs)

                existing_tags = list(wandb_init_kwargs.get("tags", []))
                if self.wandb_tag is not None:
                    existing_tags.append(self.wandb_tag)

                run = wandb.init(
                    **wandb_init_kwargs,
                    config=params,
                    name=f"{self.study_name}_{trial.number}",
                    group=self.study_name,
                    tags=existing_tags,
                    save_code=True,
                    reinit=True,
                )

            threads = max(1, self.n_cpus // self.n_gpus)
            mp_ctx = get_context("spawn")
            per_seed_aggregates: List[float] = []

            for seed in range(num_seeds):
                MAIN_LOGGER.info(f"[TRIAL {trial.number}] Seed {seed} starting")
                shards = _round_robin_split(env_ids, self.n_gpus)
                pools, results = [], []

                def _on_done(res_tuple):
                    with self._progress_lock:
                        self._runs_completed += 1
                        elapsed = _hhmmss(time.time() - self._tune_start_ts)
                        env = res_tuple[0] if isinstance(res_tuple, (list, tuple)) and res_tuple else "?"
                        self._append_trk(
                            f"done {self._runs_completed}/{self._runs_total} | elapsed {elapsed} | "
                            f"trial {trial.number} seed {seed} env {env}"
                        )

                def _on_err(exc):
                    with self._progress_lock:
                        elapsed = _hhmmss(time.time() - self._tune_start_ts)
                        self._append_trk(
                            f"error at {self._runs_completed}/{self._runs_total} | elapsed {elapsed} | "
                            f"trial {trial.number} seed {seed} | {repr(exc)}"
                        )
                    MAIN_LOGGER.error(f"[TRIAL {trial.number}] Worker error:\n{traceback.format_exc()}")

                try:
                    for gpu_id in self.gpus:
                        p = mp_ctx.Pool(
                            processes=1,
                            initializer=_init_worker,
                            initargs=(gpu_id, threads, self.study_name, self.logs_root, level),
                        )
                        pools.append(p)

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

                    for ar in async_results:
                        try:
                            res = ar.get()
                        except Exception:
                            raise

                finally:
                    for p in pools:
                        try: p.close()
                        except Exception: pass
                    for p in pools:
                        try: p.join()
                        except Exception: pass

                if not results:
                    MAIN_LOGGER.error(f"[TRIAL {trial.number}] No results for seed {seed} — aborting.")
                    raise RuntimeError("Worker failure; terminating study.")

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
                MAIN_LOGGER.info(f"Trials completed: {len(study.trials)}; Best={getattr (study.best_trial, 'value', None)}")
            else:
                MAIN_LOGGER.info("No trials completed.")
        MAIN_LOGGER.info(f"BEST TRIAL: value={study.best_trial.value} params={study.best_trial.params}")
        return study.best_trial
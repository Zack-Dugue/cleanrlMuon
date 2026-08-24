"""Subprocess and reporting helpers for Craftax PQN experiments."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

Device = Union[int, str]

# Craftax's published score is the sum of one point per achievement. These
# denominators make optimizer tuning comparable when both symbolic variants
# are requested. Values are intentionally not clipped at one.
MAX_ACHIEVEMENT_SCORE = {
    "Craftax-Classic-Symbolic-v1": 22.0,
    "Craftax-Symbolic-v1": 226.0,
}


def safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value).strip())
    return slug[:160] or "none"


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_envs(value: str) -> List[str]:
    envs = parse_csv(value)
    if not envs:
        raise ValueError("At least one comma-separated Craftax environment is required")
    return envs


def parse_devices(value: str) -> List[Device]:
    devices: List[Device] = []
    for item in parse_csv(value):
        devices.append("cpu" if item.lower() == "cpu" else int(item))
    if not devices:
        raise ValueError("At least one GPU index or 'cpu' is required")
    if "cpu" in devices and len(devices) > 1:
        raise ValueError("Use either CPU or one or more GPU indices, not both")
    return devices


def params_to_argv(params: Mapping[str, object]) -> List[str]:
    argv: List[str] = []
    for key, value in params.items():
        if value is None:
            continue
        normalized = str(key).replace("_", "-")
        flag = "--" + normalized
        if isinstance(value, bool):
            argv.append(flag if value else "--no-" + normalized)
        else:
            argv.extend((flag, str(value)))
    return argv


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "(log was not created)"
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def normalized_return(score: float, env_id: str) -> float:
    if env_id not in MAX_ACHIEVEMENT_SCORE:
        raise ValueError(
            f"No normalization denominator is registered for {env_id!r}; "
            "add it to MAX_ACHIEVEMENT_SCORE or tune a supported symbolic environment."
        )
    return float(score) / MAX_ACHIEVEMENT_SCORE[env_id]


def run_training(
    *,
    script: Path,
    params: Mapping[str, object],
    env_id: str,
    seed: int,
    device: Device,
    output_dir: Path,
    log_path: Path,
    track: bool,
    wandb_project_name: str,
    wandb_entity: Optional[str],
    wandb_group: Optional[str],
    cpus_per_run: int,
    dry_run: bool,
    extra_args: Sequence[str] = (),
) -> Dict[str, object]:
    effective_params = dict(params)
    effective_params.update(
        {
            "env-id": env_id,
            "seed": int(seed),
            "output-dir": str(output_dir),
            "track": bool(track),
        }
    )
    if track:
        effective_params.update(
            {
                "wandb-project-name": wandb_project_name,
                "wandb-entity": wandb_entity,
                "wandb-group": wandb_group,
            }
        )
    if device == "cpu":
        effective_params["cuda"] = False

    command = [
        sys.executable,
        "-u",
        str(script),
        *params_to_argv(effective_params),
        *extra_args,
    ]
    command_text = " ".join(subprocess.list2cmdline([argument]) for argument in command)
    if dry_run:
        return {
            "status": "dry_run",
            "env_id": env_id,
            "seed": seed,
            "optimizer": params.get("optimizer"),
            "device": device,
            "command": command_text,
            "output_dir": str(output_dir),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(cpus_per_run)
    environment["MKL_NUM_THREADS"] = str(cpus_per_run)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
    if device != "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = str(device)

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"COMMAND: {command_text}\n\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(script.parents[1]),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    result: Dict[str, object] = {
        "env_id": env_id,
        "seed": int(seed),
        "optimizer": params.get("optimizer"),
        "device": device,
        "returncode": completed.returncode,
        "launcher_seconds": time.time() - started,
        "command": command_text,
        "log_path": str(log_path),
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "summary.json"
    if completed.returncode == 0 and summary_path.exists():
        result.update(read_json(summary_path))
        result["status"] = "ok"
        return result
    result["status"] = "failed"
    result["error"] = tail(log_path)
    return result

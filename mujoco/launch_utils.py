"""Shared subprocess and reporting helpers for MuJoCo PQN experiments."""

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


DEFAULT_SCORE_RANGES: Dict[str, List[float]] = {
    "HalfCheetah-v4": [-500.0, 12_000.0],
    "Hopper-v4": [0.0, 3_500.0],
    "Walker2d-v4": [0.0, 5_000.0],
    "Ant-v4": [0.0, 6_000.0],
    "Humanoid-v4": [0.0, 6_000.0],
    "Swimmer-v4": [0.0, 360.0],
    "Reacher-v4": [-50.0, 0.0],
    "Pusher-v4": [-150.0, 0.0],
}
for _environment, _score_range in list(DEFAULT_SCORE_RANGES.items()):
    DEFAULT_SCORE_RANGES[_environment.replace("-v4", "-v5")] = list(_score_range)


def safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value).strip())
    return slug[:160] or "none"


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_devices(value: str) -> List[Device]:
    devices: List[Device] = []
    for item in parse_csv(value):
        if item.lower() == "cpu":
            devices.append("cpu")
        else:
            devices.append(int(item))
    if not devices:
        raise ValueError("At least one GPU index or 'cpu' is required")
    if "cpu" in devices and len(devices) > 1:
        raise ValueError("Use either CPU or one or more GPU indices, not both")
    return devices


def params_to_argv(params: Mapping[str, object]) -> List[str]:
    argv: List[str] = []
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            argv.append(flag if value else "--no-" + key.replace("_", "-"))
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


def normalized_score(
    env_id: str, score: float, score_ranges: Mapping[str, Sequence[float]]
) -> float:
    if env_id not in score_ranges:
        raise KeyError(
            f"No score range for {env_id}. Add it with --score-ranges-json or choose a default MuJoCo task."
        )
    low, high = score_ranges[env_id]
    if high <= low:
        raise ValueError(f"Invalid score range for {env_id}: [{low}, {high}]")
    return (float(score) - float(low)) / (float(high) - float(low))


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
        {"env-id": env_id, "seed": seed, "output-dir": str(output_dir)}
    )
    if track:
        effective_params.update(
            {
                "track": True,
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
        "seed": seed,
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

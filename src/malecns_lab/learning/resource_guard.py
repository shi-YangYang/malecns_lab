"""Parent-process wall/RSS supervision; sampled limits are not an OS sandbox."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

GIB = 1024 ** 3
DEFAULT_SYSTEM_RESERVE_BYTES = GIB


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_status(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {"status": "malformed"}
    except (OSError, ValueError):
        return {"status": "missing_or_malformed"}


def resource_budget(available_bytes: int, *, rss_bytes: int = 6 * GIB,
                    reserve_bytes: int = DEFAULT_SYSTEM_RESERVE_BYTES) -> dict:
    if min(available_bytes, rss_bytes, reserve_bytes) < 0 or rss_bytes == 0:
        raise ValueError("Resource budgets must be nonnegative with a positive RSS limit")
    limit = min(rss_bytes, max(0, available_bytes - reserve_bytes))
    return {"available_bytes_before_launch": available_bytes, "requested_rss_limit_bytes": rss_bytes,
            "reserve_bytes": reserve_bytes, "effective_rss_limit_bytes": limit,
            "can_launch": limit > 0}


def supervise(command: list[str], directory: Path, *, rss_limit_bytes: int,
              wall_seconds: float = 600, reserve_bytes: int = DEFAULT_SYSTEM_RESERVE_BYTES,
              poll_seconds: float = 0.1) -> dict:
    """Report process completion or supervision failure, never scientific success.

    A clean exit yields ``completed``. The caller must independently validate
    domain evidence before promoting that result to ``passed``.
    """
    if (rss_limit_bytes <= 0 or reserve_bytes < 0 or not math.isfinite(wall_seconds)
            or wall_seconds <= 0 or not 0 < poll_seconds <= 1):
        raise ValueError("Invalid supervisor budget")
    started = time.monotonic()
    child = None
    result = {"schema": "cnsbridge.resource_run.v1", "status": "failed", "reason": None,
              "peak_observed_rss_bytes": 0, "wall_seconds": 0.0, "returncode": None}
    try:
        with (directory / "worker.log").open("w") as log, (directory / "resources.csv").open("w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["wall_seconds", "tree_rss_bytes", "system_available_bytes"])
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            process = psutil.Process(child.pid)
            while True:
                elapsed = time.monotonic() - started
                rss = 0
                try:
                    for item in [process, *process.children(recursive=True)]:
                        try:
                            rss += item.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                available = psutil.virtual_memory().available
                result["peak_observed_rss_bytes"] = max(result["peak_observed_rss_bytes"], rss)
                writer.writerow([elapsed, rss, available])
                csvfile.flush()
                if elapsed > wall_seconds:
                    result["reason"] = "wall_limit"
                    break
                if rss > rss_limit_bytes:
                    result["reason"] = "rss_limit"
                    break
                if available < reserve_bytes:
                    result["reason"] = "system_reserve"
                    break
                if child.poll() is not None:
                    result["reason"] = "worker_exited"
                    break
                time.sleep(poll_seconds)
    except BaseException as error:
        result["reason"] = f"supervisor_exception:{type(error).__name__}:{error}"
    finally:
        if child is not None:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=2)
                except ProcessLookupError:
                    pass
            result["returncode"] = child.wait()
        result["wall_seconds"] = time.monotonic() - started
        worker = _read_status(directory / "worker_result.json")
        result["worker_status"] = worker.get("status", "missing")
        phase_path = directory / "phase.json"
        result["last_phase"] = _read_status(phase_path) if phase_path.exists() else None
        if result["reason"] == "worker_exited" and result["returncode"] == 0:
            result["status"] = "completed"
        write_json(directory / "result.json", result)
    return result

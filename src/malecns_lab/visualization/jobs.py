"""Single-worker experiment ownership, memory/time limits and durable state."""

from __future__ import annotations

import copy
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil

from ..paths import project_asset_path
from .export import DEFAULT_ANNOTATIONS, DEFAULT_GRAPH
from .parameters import total_frames, validate_parameters
from .storage import DISPLAY_ASSETS, EVIDENCE_FILES, read_json, write_json

GIB = 1024**3
TERMINAL = {"completed", "failed", "cancelled"}


class JobError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def valid_job_id(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, AttributeError):
        return False


class JobManager:
    def __init__(self, root: Path | None = None, *, graph: Path | None = None, annotations: Path | None = None,
                 reserve_bytes: int = GIB, minimum_worker_bytes: int = int(1.25 * GIB), budget_s: float = 300,
                 available_memory=None, command_factory=None):
        self.root = (root or project_asset_path("data/atlas_experiments")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.graph = (graph or project_asset_path(DEFAULT_GRAPH)).resolve()
        self.annotations = (annotations or project_asset_path(DEFAULT_ANNOTATIONS)).resolve()
        self.reserve_bytes, self.minimum_worker_bytes, self.budget_s = reserve_bytes, minimum_worker_bytes, budget_s
        self.available_memory = available_memory or (lambda: psutil.virtual_memory().available)
        # Injection is for isolated process-lifecycle tests. CLI has no such option.
        self.command_factory = command_factory
        self.lock = threading.RLock()
        self.jobs, self.active_id, self.process, self.started = {}, None, None, None
        self.closed = False
        self.stop_event = threading.Event()
        for directory in self.root.iterdir():
            if directory.is_symlink() or not directory.is_dir() or not valid_job_id(directory.name):
                continue
            try:
                job = read_json(directory / "job.json")
                if job["id"] != directory.name:
                    continue
                # Historical experiments outside the current stimulus interface
                # remain on disk without recovering or rewriting their records.
                validate_parameters(job["parameters"])
                if job["status"] not in TERMINAL:
                    self._recover_process(job, directory)
                    job.update(status="failed", error="上次实验因服务中断而停止，请重新运行。", catalog_url=None)
                    write_json(directory / "job.json", job)
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError, TypeError):
                continue
        self.monitor = threading.Thread(target=self._monitor, name="atlas-job-monitor", daemon=True)
        self.monitor.start()

    @staticmethod
    def _recover_process(job: dict, directory: Path) -> None:
        """Only reclaim our exact persisted worker, never a reused/unrelated PID."""
        try:
            owner = psutil.Process(job["owner_pid"])
            if owner.is_running() and owner.create_time() == job["owner_created_at"]:
                raise JobError("已有实验服务管理此目录，请关闭重复服务。", 409)
        except (KeyError, psutil.NoSuchProcess):
            pass
        try:
            process = psutil.Process(job["worker_pid"])
            command = process.cmdline()
            if (process.create_time() == job["worker_created_at"] and "malecns_lab.visualization.worker" in command
                    and str(directory) in command):
                process.terminate()
                try:
                    process.wait(timeout=1)
                except psutil.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        except (KeyError, psutil.Error):
            pass

    def health(self) -> dict:
        with self.lock:
            return {"status": "ok", "pid": os.getpid(), "active_job_id": self.active_id,
                    "available_memory_bytes": int(self.available_memory()), "reserve_bytes": self.reserve_bytes,
                    "minimum_worker_bytes": self.minimum_worker_bytes, "budget_s": self.budget_s}

    def list(self) -> list[dict]:
        with self.lock:
            return copy.deepcopy(sorted(self.jobs.values(), key=lambda row: row["created_at"], reverse=True))

    def get(self, job_id: str) -> dict:
        with self.lock:
            if not valid_job_id(job_id) or job_id not in self.jobs:
                raise JobError("实验不存在。", 404)
            return copy.deepcopy(self.jobs[job_id])

    def _persist(self, job: dict) -> None:
        write_json(self.root / job["id"] / "job.json", job)

    def create(self, parameters: dict) -> dict:
        try:
            parameters = validate_parameters(parameters)
        except ValueError as error:
            raise JobError(str(error)) from error
        with self.lock:
            if self.closed:
                raise JobError("实验服务已关闭。", 503)
            if self.active_id:
                raise JobError("已有实验正在运行，请等待完成或取消。", 409)
            if not self.graph.is_file() or not self.annotations.is_file():
                raise JobError("完整连接组或胞体坐标文件缺失，请先准备本地数据。", 503)
            available = int(self.available_memory())
            if available < self.reserve_bytes + self.minimum_worker_bytes:
                raise JobError("可用内存不足，无法在保留 1 GiB 系统内存的条件下启动完整脑实验。", 503)
            job_id = str(uuid.uuid4())
            directory = self.root / job_id
            directory.mkdir(exist_ok=False)
            job = {"id": job_id, "status": "queued", "parameters": parameters,
                   "created_at": datetime.now(timezone.utc).isoformat(), "frames_completed": 0,
                   "total_frames": total_frames(parameters), "simulation_s": 0.0,
                   "elapsed_s": 0.0, "error": None, "catalog_url": None, "rss_bytes": 0, "peak_rss_bytes": 0,
                   "rss_limit_bytes": available - self.reserve_bytes, "available_at_start_bytes": available,
                   "reserve_bytes": self.reserve_bytes, "budget_s": self.budget_s,
                   "owner_pid": os.getpid(), "owner_created_at": psutil.Process().create_time()}
            self.jobs[job_id] = job
            self.active_id, self.started = job_id, time.monotonic()
            self._persist(job)
            write_json(directory / "request.json", {"id": job_id, "parameters": parameters})
            command = (self.command_factory(directory) if self.command_factory else
                       [sys.executable, "-m", "malecns_lab.visualization.worker", "--directory", str(directory),
                        "--graph", str(self.graph), "--annotations", str(self.annotations), "--parent-pid", str(os.getpid())])
            try:
                with (directory / "worker.log").open("xb") as log:
                    self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                                    cwd=Path(__file__).parents[3])
                job["worker_pid"] = self.process.pid
                try:
                    job["worker_created_at"] = psutil.Process(self.process.pid).create_time()
                except psutil.NoSuchProcess:
                    pass
                job["status"] = "loading"
                self._persist(job)
            except OSError as error:
                self._finish(job, "failed", "无法启动实验进程，请检查本地环境。")
                write_json(directory / "launch_failure.json", {"detail": repr(error)})
            return copy.deepcopy(job)

    def _stop_process(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        # The child has its own process group, so only this task is signalled.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=2)

    def _finish(self, job: dict, status: str, error: str | None = None) -> None:
        job.update(status=status, error=error, elapsed_s=time.monotonic() - self.started,
                   catalog_url=f"/api/experiments/{job['id']}/assets/catalog.json" if status == "completed" else None)
        self._persist(job)
        self.active_id, self.process, self.started = None, None, None

    def cancel(self, job_id: str) -> dict:
        with self.lock:
            self.get(job_id)
            job = self.jobs[job_id]
            if job["status"] not in TERMINAL:
                self._stop_process()
                self._finish(job, "cancelled", "实验已取消。")
            return copy.deepcopy(job)

    def asset(self, job_id: str, filename: str) -> Path:
        with self.lock:
            job = self.get(job_id)
            if job["status"] != "completed" or filename not in DISPLAY_ASSETS:
                raise JobError("展示文件不可用。", 404)
            directory = self.root / job_id
            path = directory / filename
            if directory.is_symlink() or path.is_symlink() or not path.is_file() or path.resolve().parent != directory:
                raise JobError("展示文件不可用。", 404)
            return path

    def _check_completion(self, directory: Path, job: dict) -> dict:
        result = read_json(directory / "result.json")
        frames = job["total_frames"]
        if (result.get("status") != "completed" or result.get("frames_completed") != frames
                or result.get("total_frames") != frames or result.get("neural_steps") != frames * 100
                or abs(result.get("simulation_s", -1) - frames / 20) > 1e-8
                or result.get("coverage", {}).get("valid_input_frames") != frames):
            raise ValueError("Incomplete worker result")
        for name in (*DISPLAY_ASSETS, *EVIDENCE_FILES):
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size == 0 or name not in result.get("artifacts", {}):
                raise ValueError("Missing worker artifact")
        return result

    def _tick(self) -> None:
        with self.lock:
            if self.active_id is None:
                return
            job, process = self.jobs[self.active_id], self.process
            directory = self.root / self.active_id
            job["elapsed_s"] = time.monotonic() - self.started
            available = int(self.available_memory())
            job["available_memory_bytes"] = available
            try:
                rss = psutil.Process(process.pid).memory_info().rss
                job["rss_bytes"] = rss
                job["peak_rss_bytes"] = max(job["peak_rss_bytes"], rss)
            except psutil.NoSuchProcess:
                pass
            reason = None
            if job["elapsed_s"] >= self.budget_s:
                reason = "实验超过 300 秒计算预算，已停止。"
            elif available < self.reserve_bytes or job["rss_bytes"] > job["rss_limit_bytes"]:
                reason = "可用内存不足，实验已停止并保留已有记录。"
            if reason:
                self._stop_process()
                self._finish(job, "failed", reason)
                return
            try:
                progress = read_json(directory / "progress.json")
                count = progress["frames_completed"]
                if (progress["status"] in ("loading", "running", "exporting") and type(count) is int
                        and job["frames_completed"] <= count <= job["total_frames"]
                        and abs(progress["simulation_s"] - count / 20) < 1e-8):
                    job.update(status=progress["status"], frames_completed=count, simulation_s=progress["simulation_s"])
            except (OSError, ValueError, KeyError, TypeError):
                pass
            code = process.poll()
            if code is not None:
                if code == 0:
                    try:
                        result = self._check_completion(directory, job)
                        job.update(frames_completed=result["frames_completed"], simulation_s=result["simulation_s"],
                                   graph=result.get("graph"), model_source_id=result.get("model_source_id"))
                        self._finish(job, "completed")
                    except (OSError, ValueError, KeyError, TypeError):
                        self._finish(job, "failed", "实验结果不完整，未发布展示文件。")
                else:
                    message = "实验计算失败，请查看本地实验记录。"
                    try:
                        message = read_json(directory / "failure.json")["error"]
                    except (OSError, ValueError, KeyError):
                        pass
                    self._finish(job, "failed", message)
            else:
                self._persist(job)

    def _monitor(self) -> None:
        while not self.stop_event.wait(0.2):
            try:
                self._tick()
            except Exception:
                # Monitoring failure must not leave an unbounded computation.
                with self.lock:
                    if self.active_id:
                        job = self.jobs[self.active_id]
                        self._stop_process()
                        self._finish(job, "failed", "资源监控异常，实验已停止。")

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.active_id:
                self.cancel(self.active_id)
            self.stop_event.set()
        self.monitor.join(timeout=3)

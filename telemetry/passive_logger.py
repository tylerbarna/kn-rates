"""Passive in-job resource monitor and execution hook."""

from contextlib import AbstractContextManager
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Dict, Optional

from telemetry.harvester import parse_memory_to_gb
from telemetry.models import JobTelemetryRecord


def get_peak_rss_gb() -> float:
    """Return maximum resident set size (RSS) consumed by the process in GB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in kilobytes on Linux, but bytes on macOS/Darwin
    if platform.system() == "Darwin":
        return round(usage.ru_maxrss / (1024.0 * 1024.0 * 1024.0), 4)
    else:
        return round(usage.ru_maxrss / (1024.0 * 1024.0), 4)


def get_cpu_seconds() -> float:
    """Return total CPU time (user + system) spent by the process in seconds."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return round(usage.ru_utime + usage.ru_stime, 2)


def get_path_size_mb(path: Optional[os.PathLike]) -> float:
    """Compute the disk usage of a file or directory in megabytes."""
    if not path or not os.path.exists(path):
        return 0.0
    p = Path(path)
    if p.is_file():
        return round(p.stat().st_size / (1024.0 * 1024.0), 3)
    total_bytes = sum(f.stat().st_size for f in p.glob("**/*") if f.is_file())
    return round(total_bytes / (1024.0 * 1024.0), 3)


class JobResourceTracker(AbstractContextManager):
    """Context manager for passively instrumenting job resource consumption and parameters."""

    def __init__(
        self,
        job_name: Optional[str] = None,
        suite: Optional[str] = None,
        model: Optional[str] = None,
        n_transients: int = 0,
        chunk_size: int = 1,
        output_path: Optional[os.PathLike] = None,
        events_file: Optional[Path] = None,
    ):
        self.job_name = job_name or os.environ.get("SLURM_JOB_NAME", "unnamed_job")
        self.suite = suite or ""
        self.model = model or ""
        self.n_transients = n_transients
        self.chunk_size = chunk_size
        self.output_path = output_path
        self.events_file = events_file or (Path.cwd() / "telemetry" / "job_events.jsonl")
        
        self.job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        self.array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        self.partition = os.environ.get("SLURM_JOB_PARTITION", "msismall")
        self.req_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        
        raw_mem = os.environ.get("SLURM_MEM_PER_NODE") or os.environ.get("SLURM_MEM_PER_CPU", "0")
        self.req_mem_gb = parse_memory_to_gb(raw_mem)

        self._t0: float = 0.0
        self._start_iso: str = ""

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._start_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._t0
        end_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        peak_rss = get_peak_rss_gb()
        cpu_sec = get_cpu_seconds()
        storage_mb = get_path_size_mb(self.output_path)
        state = "FAILED" if exc_type is not None else "COMPLETED"
        exit_code = "1:0" if exc_type is not None else "0:0"

        # Slurm submit time if available via scontrol or env
        submit_iso = os.environ.get("SLURM_SUBMIT_TIME", self._start_iso)

        rec = JobTelemetryRecord(
            job_id=self.job_id,
            array_task_id=self.array_task_id,
            job_name=self.job_name,
            suite=self.suite,
            partition=self.partition,
            submit_time=submit_iso,
            start_time=self._start_iso,
            end_time=end_iso,
            req_cpus=self.req_cpus,
            req_mem_gb=self.req_mem_gb,
            used_walltime_sec=round(elapsed, 2),
            used_max_rss_gb=peak_rss,
            used_cpu_sec=cpu_sec,
            model=self.model,
            n_transients=self.n_transients,
            chunk_size=self.chunk_size,
            output_storage_mb=storage_mb,
            state=state,
            exit_code=exit_code,
            data_source="passive_hook",
        )
        rec.compute_derived_metrics()

        # Atomic append to telemetry events file
        try:
            self.events_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict()) + "\n")
        except Exception as e:
            # Passive monitoring must never crash the primary computation
            sys.stderr.write(f"Warning: Failed to append telemetry record: {e}\n")

        # Do not suppress original exceptions
        return False

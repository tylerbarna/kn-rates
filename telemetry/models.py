"""Data models and schemas for job resource and queue telemetry tracking."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import math
import re
from typing import Any, Dict, Optional


@dataclass
class JobTelemetryRecord:
    """Represents a standardized execution record for an HPC / Slurm job."""

    job_id: str
    array_task_id: str = ""
    job_name: str = ""
    suite: str = ""
    partition: str = "msismall"
    node_list: str = ""
    
    # Scheduling & Timings
    submit_time: str = ""
    start_time: str = ""
    end_time: str = ""
    submit_hour: int = -1
    submit_day_of_week: str = ""
    is_weekend: bool = False
    concurrent_tasks: int = 1
    queue_wait_sec: float = 0.0
    queue_wait_min: float = 0.0
    queue_wait_hours: float = 0.0

    # Resource Allocations & Consumption
    req_cpus: int = 1
    req_mem_gb: float = 0.0
    req_time_min: float = 0.0
    used_walltime_sec: float = 0.0
    used_walltime_hours: float = 0.0
    used_max_rss_gb: float = 0.0
    mem_efficiency_pct: float = 0.0
    used_cpu_sec: float = 0.0
    cpu_efficiency_pct: float = 0.0

    # Astrophysical Simulation Parameters & Footprint
    model: str = ""
    n_transients: int = 0
    chunk_size: int = 1
    output_storage_mb: float = 0.0

    # Job State
    state: str = "COMPLETED"
    exit_code: str = "0:0"
    data_source: str = "sacct"

    def compute_derived_metrics(self) -> None:
        """Derive queue wait times, calendar features, and resource efficiencies."""
        dt_submit = self._parse_iso(self.submit_time)
        dt_start = self._parse_iso(self.start_time)
        dt_end = self._parse_iso(self.end_time)

        if dt_submit:
            self.submit_hour = dt_submit.hour
            self.submit_day_of_week = dt_submit.strftime("%a")
            self.is_weekend = dt_submit.weekday() >= 5

        if dt_submit and dt_start:
            wait = (dt_start - dt_submit).total_seconds()
            self.queue_wait_sec = max(0.0, float(wait))
            self.queue_wait_min = round(self.queue_wait_sec / 60.0, 2)
            self.queue_wait_hours = round(self.queue_wait_sec / 3600.0, 3)

        if dt_start and dt_end and self.used_walltime_sec <= 0:
            wall = (dt_end - dt_start).total_seconds()
            self.used_walltime_sec = max(0.0, float(wall))

        if self.used_walltime_sec > 0:
            self.used_walltime_hours = round(self.used_walltime_sec / 3600.0, 3)

        if self.req_mem_gb > 0 and self.used_max_rss_gb > 0:
            eff = (self.used_max_rss_gb / self.req_mem_gb) * 100.0
            self.mem_efficiency_pct = round(min(100.0, eff), 2)

        if self.used_walltime_sec > 0 and self.req_cpus > 0 and self.used_cpu_sec > 0:
            cpu_eff = (self.used_cpu_sec / (self.used_walltime_sec * self.req_cpus)) * 100.0
            self.cpu_efficiency_pct = round(min(100.0, cpu_eff), 2)

    @staticmethod
    def _parse_iso(val: str) -> Optional[datetime]:
        if not val or val in ("Unknown", "N/A", "NONE"):
            return None
        # Handle ISO strings with T separator e.g. 2026-10-20T20:41:25
        clean = re.sub(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})", r"\1 \2", val)
        
        # If string contains timezone code (e.g. CDT, CST, EST, EDT, UTC)
        tokens = clean.split()
        if len(tokens) == 6:
            # e.g., 'Tue', 'Oct', '20', '20:41:25', 'CDT', '2026'
            try:
                clean_tz_stripped = f"{tokens[0]} {tokens[1]} {tokens[2]} {tokens[3]} {tokens[5]}"
                return datetime.strptime(clean_tz_stripped, "%a %b %d %H:%M:%S %Y")
            except ValueError:
                pass

        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%a %b %d %H:%M:%S %Y"):
            try:
                return datetime.strptime(clean, fmt)
            except ValueError:
                continue
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a flat dictionary for tabular CSV/JSON serialization."""
        return asdict(self)


@dataclass
class RightsizingRecommendation:
    """Predictive recommendation for allocating resources to an upcoming simulation."""

    model: str
    n_transients: int
    recommended_mem_gb: float
    recommended_walltime_str: str
    recommended_walltime_sec: float
    recommended_cpus: int
    recommended_partition: str
    est_peak_ram_gb: float
    est_walltime_sec: float
    est_storage_mb: float
    queue_advice: str
    confidence_notes: str

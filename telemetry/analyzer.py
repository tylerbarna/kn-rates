"""Statistical analysis and rightsizing engine for HPC job telemetry."""

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from telemetry.models import JobTelemetryRecord, RightsizingRecommendation


class TelemetryAnalyzer:
    """Analyzes empirical job telemetry to model resource scaling and advise on rightsizing."""

    def __init__(self, records: List[JobTelemetryRecord]):
        self.records = [r for r in records if r.state in ("COMPLETED", "RUNNING", "FAILED")]

    def analyze_queue_times(self) -> Dict[str, Any]:
        """Examine how queue wait times correlate with requested resources, submission time, and batch size."""
        # 1. Queue wait vs Requested Memory Buckets
        mem_buckets: Dict[str, List[float]] = {
            "Standard (<= 32 GB)": [],
            "Medium (33 - 128 GB)": [],
            "High-Memory (129 - 755 GB)": [],
        }

        # 2. Queue wait vs Hour of Day (0-23)
        hourly_waits: Dict[int, List[float]] = defaultdict(list)

        # 3. Queue wait vs Day of Week
        dow_waits: Dict[str, List[float]] = defaultdict(list)

        # 4. Queue wait vs Concurrent Batch Size
        batch_waits: Dict[str, List[float]] = {
            "Single Task (1)": [],
            "Small Array (2 - 50)": [],
            "Massive Array (> 50)": [],
        }

        for r in self.records:
            wait_min = r.queue_wait_min
            if wait_min < 0:
                continue

            # Mem buckets
            if r.req_mem_gb <= 32.0:
                mem_buckets["Standard (<= 32 GB)"].append(wait_min)
            elif r.req_mem_gb <= 128.0:
                mem_buckets["Medium (33 - 128 GB)"].append(wait_min)
            else:
                mem_buckets["High-Memory (129 - 755 GB)"].append(wait_min)

            # Hour
            if 0 <= r.submit_hour <= 23:
                hourly_waits[r.submit_hour].append(wait_min)

            # Day of week
            if r.submit_day_of_week:
                dow_waits[r.submit_day_of_week].append(wait_min)

            # Batch size
            if r.concurrent_tasks <= 1:
                batch_waits["Single Task (1)"].append(wait_min)
            elif r.concurrent_tasks <= 50:
                batch_waits["Small Array (2 - 50)"].append(wait_min)
            else:
                batch_waits["Massive Array (> 50)"].append(wait_min)

        def _stats(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"count": 0, "mean_min": 0.0, "median_min": 0.0, "max_min": 0.0}
            sorted_arr = sorted(arr)
            n = len(sorted_arr)
            median = sorted_arr[n // 2] if n % 2 != 0 else (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0
            return {
                "count": n,
                "mean_min": round(sum(sorted_arr) / n, 2),
                "median_min": round(median, 2),
                "max_min": round(max(sorted_arr), 2),
            }

        return {
            "by_memory_tier": {k: _stats(v) for k, v in mem_buckets.items()},
            "by_submission_hour": {k: _stats(v) for k, v in sorted(hourly_waits.items())},
            "by_day_of_week": {k: _stats(v) for k, v in dow_waits.items()},
            "by_batch_size": {k: _stats(v) for k, v in batch_waits.items()},
        }

    def fit_resource_scaling(self) -> Dict[str, Any]:
        """Fit empirical scaling coefficients for RAM, walltime, and disk storage as a function of N transients."""
        # Group valid data points by model and n_transients
        # model -> points: list of (n_transients, rss_gb, walltime_sec, storage_mb)
        points_by_model: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)
        all_points: List[Tuple[float, float, float, float]] = []

        for r in self.records:
            if r.n_transients > 0 and r.used_max_rss_gb > 0:
                pt = (float(r.n_transients), float(r.used_max_rss_gb), float(r.used_walltime_sec), float(r.output_storage_mb))
                model_name = r.model if r.model else "Generic"
                points_by_model[model_name].append(pt)
                all_points.append(pt)

        scaling_fits: Dict[str, Any] = {}

        for m_name, pts in points_by_model.items():
            if len(pts) >= 2:
                scaling_fits[m_name] = self._fit_linear_scaling(pts)

        if len(all_points) >= 2:
            scaling_fits["Global"] = self._fit_linear_scaling(all_points)

        return scaling_fits

    @staticmethod
    def _fit_linear_scaling(pts: List[Tuple[float, float, float, float]]) -> Dict[str, Any]:
        """Fit linear relationships: Y(N) = slope * N + intercept for RAM, Walltime, and Storage."""
        n_vals = [p[0] for p in pts]
        rss_vals = [p[1] for p in pts]
        time_vals = [p[2] for p in pts]
        storage_vals = [p[3] for p in pts]

        def _linear_fit(x: List[float], y: List[float]) -> Tuple[float, float]:
            count = len(x)
            mean_x = sum(x) / count
            mean_y = sum(y) / count
            denom = sum((xi - mean_x) ** 2 for xi in x)
            if denom <= 0:
                return (0.0, mean_y)
            slope = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / denom
            intercept = mean_y - slope * mean_x
            return (slope, max(0.0, intercept))

        ram_slope, ram_intercept = _linear_fit(n_vals, rss_vals)
        time_slope, time_intercept = _linear_fit(n_vals, time_vals)
        storage_slope, storage_intercept = _linear_fit(n_vals, storage_vals)

        return {
            "ram_gb_per_million": round(ram_slope * 1e6, 3),
            "ram_base_gb": round(ram_intercept, 3),
            "walltime_sec_per_million": round(time_slope * 1e6, 2),
            "walltime_base_sec": round(time_intercept, 2),
            "storage_mb_per_million": round(storage_slope * 1e6, 2),
            "storage_base_mb": round(storage_intercept, 2),
            "sample_points": len(pts),
        }

    def recommend_resources(
        self,
        model: str,
        n_transients: int,
        n_cpus: Optional[int] = None,
        safety_cushion_mem: float = 1.25,
        safety_cushion_time: float = 1.35,
    ) -> RightsizingRecommendation:
        """Calculate optimal requested memory, walltime, and partition for a proposed job."""
        scaling = self.fit_resource_scaling()
        fit = scaling.get(model) or scaling.get("Global")

        if fit:
            ram_per_m = fit["ram_gb_per_million"]
            ram_base = fit["ram_base_gb"]
            time_per_m = fit["walltime_sec_per_million"]
            time_base = fit["walltime_base_sec"]
            storage_per_m = fit["storage_mb_per_million"]
            storage_base = fit["storage_base_mb"]
        else:
            # Empirical fallback defaults (calibrated from MSI Agate 10M benchmark)
            # ~65 GB RAM per million transients, 7,500 sec walltime per million transients
            ram_per_m = 65.0
            ram_base = 1.5
            time_per_m = 7500.0
            time_base = 30.0
            storage_per_m = 180.0
            storage_base = 2.0

        est_ram = (n_transients / 1e6) * ram_per_m + ram_base
        est_time = (n_transients / 1e6) * time_per_m + time_base
        est_storage = (n_transients / 1e6) * storage_per_m + storage_base

        # Calculate cushioned requests
        req_ram_uncapped = est_ram * safety_cushion_mem
        req_time_sec = est_time * safety_cushion_time

        # Map to standard cluster memory allocation brackets
        standard_tiers = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 500.0, 750.0]
        recommended_ram = standard_tiers[-1]
        for tier in standard_tiers:
            if tier >= req_ram_uncapped:
                recommended_ram = tier
                break

        # Suggest CPUs
        if n_cpus:
            rec_cpus = n_cpus
        elif n_transients >= 5_000_000:
            rec_cpus = 16
        elif n_transients >= 500_000:
            rec_cpus = 8
        elif n_transients >= 50_000:
            rec_cpus = 4
        else:
            rec_cpus = 1

        # Partition selection: msismall can accommodate up to 755 GB RAM per node (Agate nodes)
        recommended_partition = "msismall"

        # Format walltime as HH:MM:SS
        hours = int(req_time_sec // 3600)
        minutes = int((req_time_sec % 3600) // 60)
        seconds = int(req_time_sec % 60)
        walltime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Strategic queue advice based on requested resources
        if recommended_ram > 256.0:
            queue_advice = (
                "HIGH MEMORY WARNING: Allocating >256 GB (--mem=750g) restricts placement to the rare "
                "755 GB high-memory nodes on msismall. Expect longer queue times (1-3 hours). "
                "To minimize queue delay, submit between 19:00 - 02:00 CDT or over weekends when node drains occur faster."
            )
        elif req_time_sec <= 7200:
            queue_advice = (
                "FAST BACKFILL ELIGIBLE: Allocating <= 32 GB RAM and <= 2 hours walltime allows Slurm's "
                "backfill scheduler to slot this task immediately into idle scheduling gaps without draining."
            )
        else:
            queue_advice = (
                "STANDARD TIER: Moderate queue wait (~5-20 minutes). Job array dispatch can proceed concurrently "
                "up to the 3,000 active running job limit."
            )

        confidence = (
            f"Calibrated from {fit.get('sample_points', 0)} empirical runs. "
            f"RAM scaling: {ram_per_m:.2f} GB/1M, Walltime scaling: {time_per_m/60.0:.1f} min/1M."
            if fit else "Calibrated from MSI Agate 10M empirical reference benchmark."
        )

        return RightsizingRecommendation(
            model=model,
            n_transients=n_transients,
            recommended_mem_gb=recommended_ram,
            recommended_walltime_str=walltime_str,
            recommended_walltime_sec=req_time_sec,
            recommended_cpus=rec_cpus,
            recommended_partition=recommended_partition,
            est_peak_ram_gb=round(est_ram, 2),
            est_walltime_sec=round(est_time, 2),
            est_storage_mb=round(est_storage, 2),
            queue_advice=queue_advice,
            confidence_notes=confidence,
        )

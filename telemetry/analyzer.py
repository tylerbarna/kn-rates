"""Statistical analysis and rightsizing engine for HPC job telemetry."""

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from telemetry.models import JobTelemetryRecord, RightsizingRecommendation


class TelemetryAnalyzer:
    """Analyzes empirical job telemetry to model resource scaling and advise on rightsizing."""

    VALID_ACTIVE_STATES = (
        "COMPLETED", "RUNNING", "FAILED", "TIMEOUT",
        "OUT_OF_MEMORY", "CANCELLED", "REQUEUED", "NODE_FAIL"
    )

    def __init__(self, records: List[JobTelemetryRecord]):
        self.records = [
            r for r in records
            if any(str(r.state).startswith(st) for st in self.VALID_ACTIVE_STATES)
        ]

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

    def analyze_efficiencies(self) -> Dict[str, Any]:
        """Compute statistical distributions and waste metrics for Time, CPU, and Memory efficiencies."""
        time_effs = [r.time_efficiency_pct for r in self.records if r.time_efficiency_pct > 0]
        cpu_effs = [r.cpu_efficiency_pct for r in self.records if r.cpu_efficiency_pct > 0]
        mem_effs = [r.mem_efficiency_pct for r in self.records if r.mem_efficiency_pct > 0]

        def _eff_stats(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p25": 0.0, "p75": 0.0}
            s = sorted(arr)
            n = len(s)
            med = s[n // 2] if n % 2 != 0 else (s[n // 2 - 1] + s[n // 2]) / 2.0
            p25 = s[n // 4]
            p75 = s[(3 * n) // 4]
            return {
                "count": n,
                "mean": round(sum(s) / n, 2),
                "median": round(med, 2),
                "min": round(min(s), 2),
                "max": round(max(s), 2),
                "p25": round(p25, 2),
                "p75": round(p75, 2),
            }

        # Efficiency by job state
        by_state: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"time_eff": [], "cpu_eff": [], "mem_eff": []})
        for r in self.records:
            st = r.state.split()[0] if r.state else "UNKNOWN"
            if r.time_efficiency_pct > 0:
                by_state[st]["time_eff"].append(r.time_efficiency_pct)
            if r.cpu_efficiency_pct > 0:
                by_state[st]["cpu_eff"].append(r.cpu_efficiency_pct)
            if r.mem_efficiency_pct > 0:
                by_state[st]["mem_eff"].append(r.mem_efficiency_pct)

        state_stats = {}
        for st, metrics in by_state.items():
            state_stats[st] = {
                "time_eff": _eff_stats(metrics["time_eff"]),
                "cpu_eff": _eff_stats(metrics["cpu_eff"]),
                "mem_eff": _eff_stats(metrics["mem_eff"]),
            }

        total_wasted_core_hours = sum(r.wasted_core_hours for r in self.records)
        total_wasted_mem_gb_hours = sum(r.wasted_mem_gb_hours for r in self.records)

        return {
            "time_efficiency": _eff_stats(time_effs),
            "cpu_efficiency": _eff_stats(cpu_effs),
            "memory_efficiency": _eff_stats(mem_effs),
            "by_state": state_stats,
            "total_wasted_core_hours": round(total_wasted_core_hours, 2),
            "total_wasted_mem_gb_hours": round(total_wasted_mem_gb_hours, 2),
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

    def analyze_suite_rightsizing(self, min_jobs: int = 20) -> Dict[str, Any]:
        """Empirically rightsize simulation suites by balancing completed resource profiles against OOM failures."""
        by_suite: Dict[str, List[JobTelemetryRecord]] = defaultdict(list)
        for r in self.records:
            s = r.suite
            if s and s not in ("None", "Unknown", "Scripts", "Defunct", "Standalone / Benchmark"):
                by_suite[s].append(r)

        suite_recommendations: Dict[str, Any] = {}

        for suite_name, job_list in sorted(by_suite.items()):
            if len(job_list) < min_jobs:
                continue

            completed = [r for r in job_list if r.state.startswith("COMPLETED")]
            oom = [r for r in job_list if r.state.startswith("OUT_OF_MEMORY")]
            timeout = [r for r in job_list if r.state.startswith("TIMEOUT")]
            cancelled = [r for r in job_list if r.state.startswith("CANCELLED")]

            # Timeout analysis for walltime lower bound
            timeout_walls = [r.req_time_min * 60.0 for r in timeout if r.req_time_min > 0]
            if not timeout_walls:
                timeout_walls = [r.used_walltime_sec for r in timeout if r.used_walltime_sec > 0]
            timeout_floor_sec = max(timeout_walls) if timeout_walls else 0.0

            # Memory requested for OOM jobs
            oom_mems = [r.req_mem_gb for r in oom if r.req_mem_gb > 0]
            oom_floor = max(oom_mems) if oom_mems else 0.0

            # Completed usage stats
            comp_rss = sorted([r.used_max_rss_gb for r in completed if r.used_max_rss_gb > 0])
            n_rss = len(comp_rss)
            p50_rss = comp_rss[n_rss // 2] if n_rss > 0 else 0.0
            p90_rss = comp_rss[int(n_rss * 0.90)] if n_rss > 0 else 0.0
            p95_rss = comp_rss[int(n_rss * 0.95)] if n_rss > 0 else 0.0
            max_rss = comp_rss[-1] if n_rss > 0 else 0.0

            comp_time = sorted([r.used_walltime_sec for r in completed if r.used_walltime_sec > 0])
            n_time = len(comp_time)
            p50_time = comp_time[n_time // 2] if n_time > 0 else 0.0
            p95_time = comp_time[int(n_time * 0.95)] if n_time > 0 else 0.0
            max_time = comp_time[-1] if n_time > 0 else 0.0

            comp_cpu = [r.cpu_efficiency_pct for r in completed if r.cpu_efficiency_pct > 0]
            median_cpu = sorted(comp_cpu)[len(comp_cpu) // 2] if comp_cpu else 100.0

            # CPU core utilization analysis:
            # Effective cores used = req_cpus * (cpu_efficiency_pct / 100.0)
            comp_cores_used = [
                r.req_cpus * (r.cpu_efficiency_pct / 100.0)
                for r in completed
                if r.req_cpus > 0 and r.cpu_efficiency_pct > 0
            ]
            comp_cores_used.sort()
            n_cores = len(comp_cores_used)
            p50_cores = comp_cores_used[n_cores // 2] if n_cores > 0 else 1.0
            p95_cores = comp_cores_used[int(n_cores * 0.95)] if n_cores > 0 else p50_cores
            max_cores = comp_cores_used[-1] if n_cores > 0 else p50_cores

            # Historical requested CPUs across the suite
            req_cpus_list = [r.req_cpus for r in job_list if r.req_cpus > 0]
            hist_req_cpus = round(sum(req_cpus_list) / len(req_cpus_list)) if req_cpus_list else 4

            # Optimal memory recommendation:
            # Must strictly clear the empirical OOM floor with 25% cushion AND cover 95th percentile completed RSS
            raw_target_mem = max(oom_floor * 1.25, p95_rss * 1.25, 4.0)
            standard_tiers = [8.0, 16.0, 32.0, 64.0, 96.0, 128.0, 256.0, 500.0, 750.0]
            rec_mem = next((t for t in standard_tiers if t >= raw_target_mem), round(raw_target_mem))

            # Optimal walltime recommendation:
            # Must strictly exceed timeout ceiling (if any) and cover P95 completed elapsed time * 1.35
            rec_time_sec = max(p95_time * 1.35, timeout_floor_sec * 1.30, 300.0)
            rec_time_sec = float(int((rec_time_sec + 59) // 60) * 60)
            h = int(rec_time_sec // 3600)
            m = int((rec_time_sec % 3600) // 60)
            s_rem = int(rec_time_sec % 60)
            rec_time_str = f"{h:02d}:{m:02d}:{s_rem:02d}"

            # Optimal CPU core recommendation:
            # Cover P95 effective core demand with 25% headroom, mapped to standard core tiers (1, 2, 4, 8, 16)
            target_cores = max(p95_cores * 1.25, 1.0)
            standard_core_tiers = [1, 2, 4, 8, 16, 32, 64]
            rec_cpus = next((c for c in standard_core_tiers if c >= target_cores), 4)

            # Cluster Node Architecture & NUMA balance:
            # Prevent allocating high memory on too few cores which leads to memory bus/NUMA starvation
            if rec_mem >= 128.0 and rec_cpus < 8:
                rec_cpus = 8
            elif rec_mem >= 64.0 and rec_cpus < 4:
                rec_cpus = 4
            elif rec_mem >= 32.0 and rec_cpus < 2:
                rec_cpus = 2

            # Concurrency multiplier & Service Unit (core-hour) savings
            concurrency_mult = round(hist_req_cpus / rec_cpus, 1) if (rec_cpus > 0 and hist_req_cpus > rec_cpus) else 1.0
            su_savings_pct = round(max(0.0, (1.0 - (rec_cpus / hist_req_cpus)) * 100.0), 1) if hist_req_cpus > 0 else 0.0

            # Partition & Queue advice
            if rec_mem <= 32.0 and rec_time_sec <= 7200.0:
                partition = "msismall"
                queue_advice = "FAST BACKFILL ELIGIBLE: <=32 GB RAM and <=2hr walltime qualifies for instant Slurm backfill scheduling."
            elif rec_mem <= 128.0:
                partition = "msismall"
                queue_advice = "STANDARD COMPUTE TIER: Balanced placement on msismall standard dual-socket nodes."
            elif rec_mem <= 256.0:
                partition = "msismall"
                queue_advice = "MEDIUM MEMORY TIER: Eligible on msismall 256GB nodes (~15-45 min typical queue wait)."
            else:
                partition = "msismall"
                queue_advice = "HIGH MEMORY WARNING: Allocation >256 GB restricts scheduling to rare 755GB nodes."

            # CPU Advice
            if median_cpu < 50.0:
                cpu_advice = (
                    f"BOTTLENECK DETECTED: Historical median CPU efficiency is {median_cpu:.1f}%, with effective demand "
                    f"peaking at P95={p95_cores:.2f} cores (requested ~{hist_req_cpus} cores). Code is primarily single-threaded or I/O bound. "
                    f"Rightsizing to --cpus-per-task={rec_cpus} saves ~{su_savings_pct:.0f}% Service Units and yields "
                    f"a {concurrency_mult:.1f}x concurrency multiplier on cluster nodes."
                )
            else:
                cpu_advice = (
                    f"BALANCED CPU UTILIZATION: Historical median CPU efficiency is {median_cpu:.1f}% "
                    f"(effective P95={p95_cores:.2f} cores). Recommended --cpus-per-task={rec_cpus} ensures optimal thread scaling."
                )

            # Walltime Advice
            if timeout_floor_sec > 0:
                walltime_advice = (
                    f"TIMEOUT RISK MITIGATED: Historical timeouts occurred at <= {timeout_floor_sec/60.0:.1f} min. "
                    f"Recommended {rec_time_str} clears both the timeout ceiling and completed P95 ({p95_time/60.0:.1f} min) "
                    f"with a 35% safety buffer."
                )
            else:
                walltime_advice = (
                    f"OPTIMAL BACKFILL DURATION: Completed P95 is {p95_time/60.0:.1f} min. Recommended {rec_time_str} "
                    f"provides safety while staying short enough to minimize backfill queue wait."
                )

            # OOM risk narrative
            oom_rate = (len(oom) / len(job_list)) * 100.0 if job_list else 0.0
            if oom_floor > 0:
                risk_notes = (
                    f"Historical runs with <= {oom_floor:.0f} GB memory exhibited {len(oom):,} OOM kills "
                    f"({oom_rate:.1f}% failure rate). The recommended {rec_mem:.0f} GB allocation clears the empirical "
                    f"{max_rss:.1f} GB peak usage ceiling with a safe margin."
                )
            else:
                eff_est = (p95_rss / rec_mem * 100.0) if rec_mem > 0 else 0.0
                risk_notes = (
                    f"Zero OOM failures observed across {len(job_list)} runs. "
                    f"Recommended {rec_mem:.0f} GB provides optimal ~{eff_est:.1f}% memory efficiency."
                )

            suite_recommendations[suite_name] = {
                "sample_size": len(job_list),
                "completed_count": len(completed),
                "oom_count": len(oom),
                "timeout_count": len(timeout),
                "cancelled_count": len(cancelled),
                "oom_failure_rate_pct": round(oom_rate, 2),
                "oom_memory_floor_gb": oom_floor,
                "completed_rss_p50_gb": round(p50_rss, 2),
                "completed_rss_p95_gb": round(p95_rss, 2),
                "completed_rss_max_gb": round(max_rss, 2),
                "completed_walltime_p50_min": round(p50_time / 60.0, 2),
                "completed_walltime_p95_min": round(p95_time / 60.0, 2),
                "completed_walltime_max_min": round(max_time / 60.0, 2),
                "timeout_floor_min": round(timeout_floor_sec / 60.0, 2),
                "median_cpu_eff_pct": round(median_cpu, 2),
                "effective_cores_p50": round(p50_cores, 2),
                "effective_cores_p95": round(p95_cores, 2),
                "effective_cores_max": round(max_cores, 2),
                "historical_req_cpus": hist_req_cpus,
                "concurrency_multiplier": concurrency_mult,
                "su_savings_pct": su_savings_pct,
                "recommended_mem_gb": rec_mem,
                "recommended_walltime_sec": round(rec_time_sec, 1),
                "recommended_walltime_str": rec_time_str,
                "recommended_cpus": rec_cpus,
                "recommended_partition": partition,
                "queue_advice": queue_advice,
                "cpu_advice": cpu_advice,
                "walltime_advice": walltime_advice,
                "risk_notes": risk_notes,
            }

        return suite_recommendations

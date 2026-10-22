"""Main orchestration module and CLI for job telemetry and resource rightsizing."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from telemetry.analyzer import TelemetryAnalyzer
from telemetry.harvester import JobHarvester
from telemetry.models import JobTelemetryRecord, RightsizingRecommendation


REGISTRY_CSV = "telemetry/job_resource_registry.csv"
SCALING_MODELS_JSON = "telemetry/scaling_models.json"
EVENTS_JSONL = "telemetry/job_events.jsonl"


class JobTelemetryTracker:
    """Manages collection, persistence, and analysis of job telemetry."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.registry_path = self.project_root / REGISTRY_CSV
        self.scaling_path = self.project_root / SCALING_MODELS_JSON
        self.events_path = self.project_root / EVENTS_JSONL
        self.harvester = JobHarvester(project_root=self.project_root)

    def load_registry(self) -> List[JobTelemetryRecord]:
        """Load records from the canonical version-controlled CSV registry."""
        if not self.registry_path.exists():
            return []

        records = []
        with open(self.registry_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = JobTelemetryRecord(
                    job_id=row.get("job_id", ""),
                    array_task_id=row.get("array_task_id", ""),
                    job_name=row.get("job_name", ""),
                    suite=row.get("suite", ""),
                    partition=row.get("partition", "msismall"),
                    node_list=row.get("node_list", ""),
                    submit_time=row.get("submit_time", ""),
                    start_time=row.get("start_time", ""),
                    end_time=row.get("end_time", ""),
                    submit_hour=int(row.get("submit_hour", -1)),
                    submit_day_of_week=row.get("submit_day_of_week", ""),
                    is_weekend=row.get("is_weekend", "False").lower() in ("true", "1"),
                    concurrent_tasks=int(row.get("concurrent_tasks", 1)),
                    queue_wait_sec=float(row.get("queue_wait_sec", 0.0)),
                    queue_wait_min=float(row.get("queue_wait_min", 0.0)),
                    queue_wait_hours=float(row.get("queue_wait_hours", 0.0)),
                    req_cpus=int(row.get("req_cpus", 1)),
                    req_mem_gb=float(row.get("req_mem_gb", 0.0)),
                    req_time_min=float(row.get("req_time_min", 0.0)),
                    used_walltime_sec=float(row.get("used_walltime_sec", 0.0)),
                    used_walltime_hours=float(row.get("used_walltime_hours", 0.0)),
                    used_max_rss_gb=float(row.get("used_max_rss_gb", 0.0)),
                    mem_efficiency_pct=float(row.get("mem_efficiency_pct", 0.0)),
                    used_cpu_sec=float(row.get("used_cpu_sec", 0.0)),
                    cpu_efficiency_pct=float(row.get("cpu_efficiency_pct", 0.0)),
                    model=row.get("model", ""),
                    n_transients=int(row.get("n_transients", 0)),
                    chunk_size=int(row.get("chunk_size", 1)),
                    output_storage_mb=float(row.get("output_storage_mb", 0.0)),
                    state=row.get("state", "COMPLETED"),
                    exit_code=row.get("exit_code", "0:0"),
                    data_source=row.get("data_source", "registry"),
                )
                rec.compute_derived_metrics()
                records.append(rec)
        return records

    def update(self, start_date: str = "2026-01-01") -> List[JobTelemetryRecord]:
        """Harvest metrics, merge with registry and passive events, and write updated files."""
        existing_records = { (r.job_id, r.array_task_id): r for r in self.load_registry() }
        harvested = self.harvester.harvest_all(start_date=start_date)

        for r in harvested:
            key = (r.job_id, r.array_task_id)
            if key in existing_records:
                self.harvester._merge_records(existing_records[key], r)
            else:
                existing_records[key] = r

        # Ingest passive events if file exists
        if self.events_path.exists():
            try:
                with open(self.events_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        rec = JobTelemetryRecord(**data)
                        key = (rec.job_id, rec.array_task_id)
                        if key in existing_records:
                            self.harvester._merge_records(existing_records[key], rec)
                        else:
                            existing_records[key] = rec
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to parse events file: {e}\n")

        all_records = list(existing_records.values())
        for r in all_records:
            r.compute_derived_metrics()

        # Sort canonically: submit_time, job_id, array_task_id
        all_records.sort(key=lambda x: (x.submit_time or x.start_time or "9999", x.job_id, x.array_task_id))

        # Write to canonical CSV
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(JobTelemetryRecord("").to_dict().keys())
        with open(self.registry_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_records:
                writer.writerow(r.to_dict())

        # Fit scaling models and export JSON
        analyzer = TelemetryAnalyzer(all_records)
        scaling_fits = analyzer.fit_resource_scaling()
        queue_summary = analyzer.analyze_queue_times()

        telemetry_meta = {
            "last_updated": r.submit_time if all_records else "",
            "total_tracked_jobs": len(all_records),
            "scaling_models": scaling_fits,
            "queue_characteristics": queue_summary,
        }
        with open(self.scaling_path, "w", encoding="utf-8") as f:
            json.dump(telemetry_meta, f, indent=2)

        return all_records

    def generate_report(self) -> str:
        """Generate a formatted markdown telemetry and rightsizing summary report."""
        records = self.load_registry()
        if not records:
            records = self.update()

        analyzer = TelemetryAnalyzer(records)
        q_data = analyzer.analyze_queue_times()
        scaling = analyzer.fit_resource_scaling()

        lines = [
            "# HPC Job Telemetry, Queue Dynamics & Resource Rightsizing Registry",
            f"**Total Tracked Jobs/Tasks:** {len(records)}",
            "",
            "## 1. Queue Time Dynamics vs. Requested Resources & Scheduling",
            "",
            "### By Requested Memory Tier:",
            "| Memory Tier | Count | Median Wait | Mean Wait | Max Wait |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for tier, stats in q_data["by_memory_tier"].items():
            lines.append(f"| {tier} | {stats['count']} | {stats['median_min']:.1f} min | {stats['mean_min']:.1f} min | {stats['max_min']:.1f} min |")

        lines.extend([
            "",
            "### By Submission Hour of Day (CDT):",
            "| Hour Window | Count | Median Wait | Mean Wait |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for hr, stats in q_data["by_submission_hour"].items():
            lines.append(f"| {hr:02d}:00 - {hr:02d}:59 | {stats['count']} | {stats['median_min']:.1f} min | {stats['mean_min']:.1f} min |")

        lines.extend([
            "",
            "### By Concurrent Job Batch Size:",
            "| Batch Size Category | Count | Median Wait | Mean Wait |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for b_name, stats in q_data["by_batch_size"].items():
            lines.append(f"| {b_name} | {stats['count']} | {stats['median_min']:.1f} min | {stats['mean_min']:.1f} min |")

        lines.extend([
            "",
            "## 2. Empirical Resource Scaling Fits (per 1,000,000 Transients)",
            "| Model Family | RAM Scaling (GB/1M) | Base RAM (GB) | Walltime (min/1M) | Storage (MB/1M) | Sample Runs |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for model_name, fit in scaling.items():
            lines.append(
                f"| {model_name} | {fit['ram_gb_per_million']:.2f} GB | {fit['ram_base_gb']:.2f} GB | "
                f"{fit['walltime_sec_per_million']/60.0:.1f} min | {fit['storage_mb_per_million']:.1f} MB | {fit['sample_points']} |"
            )

        lines.extend([
            "",
            "## 3. Reference Rightsizing Recommendations",
            "| Model | Transients | Rec. Memory | Rec. Walltime | Rec. CPUs | Est. Peak RAM | Queue Advice |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        demo_cases = [
            ("Metzger", 100_000),
            ("Bu2026Fixed", 1_000_000),
            ("Bu2026Vary", 5_000_000),
            ("Bu2026Vary", 10_000_000),
        ]
        for m, n in demo_cases:
            rec = analyzer.recommend_resources(m, n)
            lines.append(
                f"| {m} | {n:,} | `--mem={int(rec.recommended_mem_gb)}g` | `--time={rec.recommended_walltime_str}` | "
                f"`--cpus-per-task={rec.recommended_cpus}` | {rec.est_peak_ram_gb:.1f} GB | {rec.queue_advice[:45]}... |"
            )

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="HPC Job Telemetry & Rightsizing Tracker.")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # Update command
    p_up = subparsers.add_parser("update", help="Harvest logs and sacct to update registry CSV")
    p_up.add_argument("--start-date", default="2026-01-01", help="Start date for sacct harvest")

    # Report command
    subparsers.add_parser("report", help="Display telemetry report and scaling fits")

    # Rightsize command
    p_right = subparsers.add_parser("rightsize", help="Calculate recommended resources for a simulation")
    p_right.add_argument("--model", default="Bu2026Fixed", choices=["Metzger", "Bu2026Fixed", "Bu2026Vary"])
    p_right.add_argument("--n-transients", type=int, default=1_000_000)
    p_right.add_argument("--cpus", type=int, default=None)

    args = parser.parse_args()
    tracker = JobTelemetryTracker()

    if args.command == "update" or args.command is None:
        recs = tracker.update(start_date=getattr(args, "start_date", "2026-01-01"))
        print(f"Updated {REGISTRY_CSV} with {len(recs)} tracked job records.")
        print(f"Scaling models updated in {SCALING_MODELS_JSON}.")
    elif args.command == "report":
        print(tracker.generate_report())
    elif args.command == "rightsize":
        analyzer = TelemetryAnalyzer(tracker.load_registry() or tracker.update())
        rec = analyzer.recommend_resources(args.model, args.n_transients, n_cpus=args.cpus)
        print("=" * 70)
        print(f"SLURM RESOURCE RIGHTSIZING RECOMMENDATION")
        print("=" * 70)
        print(f"Model:                   {rec.model}")
        print(f"Injected Transients:     {rec.n_transients:,}")
        print(f"Estimated Peak RAM:      {rec.est_peak_ram_gb:.2f} GB")
        print(f"Estimated Wall-clock:    {rec.est_walltime_sec / 60.0:.1f} minutes ({rec.est_walltime_sec / 3600.0:.2f} hours)")
        print(f"Estimated Output Size:   {rec.est_storage_mb:.1f} MB")
        print("-" * 70)
        print(f"RECOMMENDED SLURM SBATCH DIRECTIVES:")
        print(f"  #SBATCH -p {rec.recommended_partition}")
        print(f"  #SBATCH --mem={int(rec.recommended_mem_gb)}g")
        print(f"  #SBATCH --time={rec.recommended_walltime_str}")
        print(f"  #SBATCH --cpus-per-task={rec.recommended_cpus}")
        print("-" * 70)
        print(f"Queue Placement Strategy:")
        print(f"  {rec.queue_advice}")
        print(f"Confidence Notes:")
        print(f"  {rec.confidence_notes}")
        print("=" * 70)


if __name__ == "__main__":
    main()

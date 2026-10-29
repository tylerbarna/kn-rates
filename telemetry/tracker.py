"""Main orchestration module and CLI for job telemetry and resource rightsizing."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional

from telemetry.analyzer import TelemetryAnalyzer
from telemetry.harvester import JobHarvester, infer_suite_from_job_name
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
                    suite=row.get("suite", "") or infer_suite_from_job_name(row.get("job_name", "")),
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
                    req_time_min=float(row.get("req_time_min", 0.0) or 0.0),
                    used_walltime_sec=float(row.get("used_walltime_sec", 0.0) or 0.0),
                    used_walltime_hours=float(row.get("used_walltime_hours", 0.0) or 0.0),
                    time_efficiency_pct=float(row.get("time_efficiency_pct", 0.0) or 0.0),
                    used_max_rss_gb=float(row.get("used_max_rss_gb", 0.0) or 0.0),
                    mem_efficiency_pct=float(row.get("mem_efficiency_pct", 0.0) or 0.0),
                    used_cpu_sec=float(row.get("used_cpu_sec", 0.0) or 0.0),
                    cpu_efficiency_pct=float(row.get("cpu_efficiency_pct", 0.0) or 0.0),
                    energy_joules=float(row.get("energy_joules", 0.0) or 0.0),
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
        eff_summary = analyzer.analyze_efficiencies()

        telemetry_meta = {
            "last_updated": all_records[-1].submit_time if all_records else "",
            "total_tracked_jobs": len(all_records),
            "scaling_models": scaling_fits,
            "queue_characteristics": queue_summary,
            "efficiency_profile": eff_summary,
        }
        with open(self.scaling_path, "w", encoding="utf-8") as f:
            json.dump(telemetry_meta, f, indent=2)

        return all_records

    def cross_match(
        self,
        input_text: Optional[str] = None,
        input_file: Optional[Path] = None,
        user: Optional[str] = None,
        since: str = "now-3week",
    ) -> Dict[str, Any]:
        """Cross-match existing registry with reportseff data, updating efficiencies and adding new jobs."""
        reportseff_records: List[JobTelemetryRecord] = []

        if input_file and Path(input_file).exists():
            text = Path(input_file).read_text(encoding="utf-8")
            reportseff_records = self.harvester.parse_reportseff_output(text)
        elif input_text:
            reportseff_records = self.harvester.parse_reportseff_output(input_text)
        elif shutil.which("reportseff"):
            reportseff_records = self.harvester.harvest_reportseff(user=user, since=since)

        existing = { (r.job_id, r.array_task_id): r for r in self.load_registry() }
        matched_count = 0
        added_count = 0

        for r in reportseff_records:
            key = (r.job_id, r.array_task_id)
            if key in existing:
                self.harvester._merge_records(existing[key], r)
                matched_count += 1
            else:
                existing[key] = r
                added_count += 1

        all_records = list(existing.values())
        for r in all_records:
            r.compute_derived_metrics()

        all_records.sort(key=lambda x: (x.submit_time or x.start_time or "9999", x.job_id, x.array_task_id))

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(JobTelemetryRecord("").to_dict().keys())
        with open(self.registry_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_records:
                writer.writerow(r.to_dict())

        analyzer = TelemetryAnalyzer(all_records)
        scaling_fits = analyzer.fit_resource_scaling()
        queue_summary = analyzer.analyze_queue_times()
        eff_summary = analyzer.analyze_efficiencies()

        telemetry_meta = {
            "last_updated": all_records[-1].submit_time if all_records else "",
            "total_tracked_jobs": len(all_records),
            "scaling_models": scaling_fits,
            "queue_characteristics": queue_summary,
            "efficiency_profile": eff_summary,
        }
        with open(self.scaling_path, "w", encoding="utf-8") as f:
            json.dump(telemetry_meta, f, indent=2)

        return {
            "matched_updated": matched_count,
            "new_jobs_added": added_count,
            "total_registry_records": len(all_records),
            "efficiency_profile": eff_summary,
        }

    def generate_report(self) -> str:
        """Generate a formatted markdown telemetry and rightsizing summary report."""
        records = self.load_registry()
        if not records:
            records = self.update()

        analyzer = TelemetryAnalyzer(records)
        q_data = analyzer.analyze_queue_times()
        scaling = analyzer.fit_resource_scaling()
        eff_data = analyzer.analyze_efficiencies()

        lines = [
            "# HPC Job Telemetry, Queue Dynamics & Resource Rightsizing Registry",
            f"**Total Tracked Jobs/Tasks:** {len(records)}",
            "",
            "## 1. Resource Efficiency Profile (reportseff & kernel accounting)",
            "| Metric | Mean | Median | 25th % | 75th % | Evaluated Jobs |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
            f"| **Time Efficiency (`TimeEff`)** | {eff_data['time_efficiency']['mean']:.1f}% | {eff_data['time_efficiency']['median']:.1f}% | {eff_data['time_efficiency']['p25']:.1f}% | {eff_data['time_efficiency']['p75']:.1f}% | {eff_data['time_efficiency']['count']} |",
            f"| **CPU Efficiency (`CPUEff`)** | {eff_data['cpu_efficiency']['mean']:.1f}% | {eff_data['cpu_efficiency']['median']:.1f}% | {eff_data['cpu_efficiency']['p25']:.1f}% | {eff_data['cpu_efficiency']['p75']:.1f}% | {eff_data['cpu_efficiency']['count']} |",
            f"| **Memory Efficiency (`MemEff`)** | {eff_data['memory_efficiency']['mean']:.1f}% | {eff_data['memory_efficiency']['median']:.1f}% | {eff_data['memory_efficiency']['p25']:.1f}% | {eff_data['memory_efficiency']['p75']:.1f}% | {eff_data['memory_efficiency']['count']} |",
            "",
            f"- **Cumulative Wasted Core-Hours**: {eff_data['total_wasted_core_hours']:.1f} core-hours",
            f"- **Cumulative Wasted Memory Slack**: {eff_data['total_wasted_mem_gb_hours']:.1f} GB-hours",
            "",
            "## 2. Queue Time Dynamics vs. Requested Resources & Scheduling",
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

        lines.extend([
            "",
            "## 4. Empirical Rightsizing by Analysis Suite (Balancing Completed Runs & OOM Kills)",
            "| Suite Family | Sample Size | OOM Rate | OOM Mem Floor | Completed P95 RSS | Rec. Memory | Rec. Walltime | Rec. CPUs | Queue Placement |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        suite_recs = analyzer.analyze_suite_rightsizing(min_jobs=20)
        for s_name, s_data in suite_recs.items():
            lines.append(
                f"| **{s_name}** | {s_data['sample_size']:,} ({s_data['completed_count']:,} ok, {s_data['oom_count']:,} OOM) | "
                f"{s_data['oom_failure_rate_pct']:.1f}% | {s_data['oom_memory_floor_gb']:.0f} GB | "
                f"{s_data['completed_rss_p95_gb']:.1f} GB | `--mem={int(s_data['recommended_mem_gb'])}g` | "
                f"`--time={s_data['recommended_walltime_str']}` | `--cpus-per-task={s_data['recommended_cpus']}` | "
                f"{s_data['queue_advice'][:35]}... |"
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

    # Cross-match command
    p_cross = subparsers.add_parser("cross-match", help="Cross-match existing registry with reportseff data")
    p_cross.add_argument("--file", help="Path to text file containing reportseff output")
    p_cross.add_argument("--since", default="now-3week", help="Time filter for reportseff query")
    p_cross.add_argument("--user", default=None, help="User to query in reportseff")

    # Rightsize command
    p_right = subparsers.add_parser("rightsize", help="Calculate recommended resources for a simulation")
    p_right.add_argument("--model", default="Bu2026Fixed", choices=["Metzger", "Bu2026Fixed", "Bu2026Vary"])
    p_right.add_argument("--suite", default=None, help="Name of analysis suite family to rightsize")
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
    elif args.command == "cross-match":
        input_f = Path(args.file) if args.file else None
        res = tracker.cross_match(input_file=input_f, user=args.user, since=args.since)
        print("=" * 70)
        print("REPORTSEFF CROSS-MATCH SUMMARY")
        print("=" * 70)
        print(f"Matched and Updated:     {res['matched_updated']} existing job records")
        print(f"Newly Added Records:     {res['new_jobs_added']} jobs")
        print(f"Total Registry Size:     {res['total_registry_records']} jobs")
        print(f"Registry File:           {REGISTRY_CSV}")
        print(f"Scaling Models:          {SCALING_MODELS_JSON}")
        print("-" * 70)
        p = res["efficiency_profile"]
        print(f"Median Time Efficiency:  {p['time_efficiency']['median']:.1f}% ({p['time_efficiency']['count']} jobs)")
        print(f"Median CPU Efficiency:   {p['cpu_efficiency']['median']:.1f}% ({p['cpu_efficiency']['count']} jobs)")
        print(f"Median Memory Efficiency:{p['memory_efficiency']['median']:.1f}% ({p['memory_efficiency']['count']} jobs)")
        print(f"Total Wasted Core-Hours: {p['total_wasted_core_hours']:.1f} core-hrs")
        print(f"Total Memory Slack:      {p['total_wasted_mem_gb_hours']:.1f} GB-hrs")
        print("=" * 70)
    elif args.command == "rightsize":
        analyzer = TelemetryAnalyzer(tracker.load_registry() or tracker.update())
        if args.suite:
            suite_data = analyzer.analyze_suite_rightsizing(min_jobs=5)
            match = next((v for k, v in suite_data.items() if args.suite.lower() in k.lower()), None)
            suite_label = next((k for k, v in suite_data.items() if args.suite.lower() in k.lower()), args.suite)
            if not match:
                print(f"Error: No statistically significant data found for suite '{args.suite}'. Available: {list(suite_data.keys())}", file=sys.stderr)
                sys.exit(1)
            print("=" * 70)
            print(f"SLURM EMPIRICAL RIGHTSIZING: {suite_label}")
            print("=" * 70)
            print(f"Sample Size:             {match['sample_size']:,} jobs ({match['completed_count']:,} completed, {match['oom_count']:,} OOM kills)")
            print(f"Historical OOM Rate:     {match['oom_failure_rate_pct']:.1f}%")
            print(f"Historical OOM Mem Floor:{match['oom_memory_floor_gb']:.1f} GB (crashed at/below this allocation)")
            print(f"Completed Peak RAM (P95):{match['completed_rss_p95_gb']:.2f} GB (Max: {match['completed_rss_max_gb']:.2f} GB)")
            print(f"Completed Walltime (P95):{match['completed_walltime_p95_min']:.1f} minutes")
            print("-" * 70)
            print("RECOMMENDED SLURM SBATCH DIRECTIVES:")
            print(f"  #SBATCH -p {match['recommended_partition']}")
            print(f"  #SBATCH --mem={int(match['recommended_mem_gb'])}g")
            print(f"  #SBATCH --time={match['recommended_walltime_str']}")
            print(f"  #SBATCH --cpus-per-task={match['recommended_cpus']}")
            print("-" * 70)
            print("Queue Placement Strategy:")
            print(f"  {match['queue_advice']}")
            print("Risk & Failure Mitigation Basis:")
            print(f"  {match['risk_notes']}")
            print("=" * 70)
            return

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

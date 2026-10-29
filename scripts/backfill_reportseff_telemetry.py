#!/usr/bin/env python3
"""Cross-match historical telemetry registry with Slurm reportseff efficiency data.

This one-time / on-demand utility ingests reportseff outputs, reconciles them
against existing job records in telemetry/job_resource_registry.csv, overwrites
inaccurate 100% CPU efficiency artifacts from sacct CPUTimeRAW, backfills TimeEff
and MemEff, updates terminal job states, and incorporates newly discovered jobs.

Usage:
    # 1. Run with built-in MSI calibration baseline (including user cluster jobs)
    python scripts/backfill_reportseff_telemetry.py

    # 2. Pipe live reportseff output directly on MSI
    reportseff -u $USER d=now-3week | python scripts/backfill_reportseff_telemetry.py --stdin

    # 3. Ingest a saved reportseff output text file
    python scripts/backfill_reportseff_telemetry.py --input reportseff_dump.txt

    # 4. Query live reportseff binary directly if available
    python scripts/backfill_reportseff_telemetry.py --user barna314 --since now-4week
"""

import argparse
from pathlib import Path
import shutil
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telemetry.harvester import JobHarvester
from telemetry.models import JobTelemetryRecord
from telemetry.tracker import JobTelemetryTracker


# Calibration dataset provided from MSI cluster executions (User: barna314)
BUILTIN_CALIBRATION_OUTPUT = """
 JobID      State         Elapsed  TimeEff   CPUEff   MemEff 
              4374484    CANCELLED      00:00:00   0.0%      ---      0.0%  
              4477551    COMPLETED      00:00:14   0.2%      ---      0.0%  
              4477608    CANCELLED      00:00:00   0.0%      ---      0.0%  
              4478144    COMPLETED      00:00:09   15.0%     ---      0.3%  
              4733500    CANCELLED      00:00:06   10.0%    16.7%     2.4%  
              4733503    CANCELLED      00:14:59   25.0%    53.5%    19.1%  
              4734941    CANCELLED      00:00:00   0.0%      ---      0.0%  
              4735752     TIMEOUT       00:01:14  123.3%     2.7%     2.2%  
              4735813     FAILED        00:11:51   9.9%     38.5%    28.6%
              4735815    COMPLETED      00:01:25   1.2%      1.5%     0.3%  
              4954787    REQUEUED       00:01:24  140.0%     8.3%    14.9%  
              4998196    COMPLETED      00:02:46   4.6%      6.6%     3.5%  
              4998390    CANCELLED      00:00:00   0.0%      ---      0.0%  
              4998391    COMPLETED      00:00:50   0.7%      2.2%     0.5%  
              4998392    CANCELLED      00:00:00   0.0%      ---      0.0%  
              4998400    CANCELLED      00:00:00   0.0%      ---      0.0%  
              5001449    COMPLETED      00:01:33   2.6%     10.8%     2.9%  
              5001606     FAILED        00:12:15   10.2%    37.4%    15.5%  
              5001607    COMPLETED      00:00:35   0.5%      3.9%     0.4%  
              5001616    CANCELLED      00:00:00   0.0%      ---      0.0%  
              5001621    CANCELLED      00:00:00   0.0%      ---      0.0%  
              5001622    CANCELLED      00:00:00   0.0%      ---      0.0%  
              5016014    COMPLETED      00:01:43   2.9%     11.7%     3.2%  
              5016210     TIMEOUT       02:00:39  100.5%    34.8%    14.4%  
              5016211    COMPLETED      00:01:01   0.8%      1.2%     0.3%  
              5016212     FAILED        00:01:13   4.1%     17.8%    14.1%  
              5016213     FAILED        00:00:29   1.6%     13.8%     0.0%  
              5016214     FAILED        00:00:50   0.9%     10.0%     1.0%  
              5016215    CANCELLED      00:00:00   0.0%      ---      0.0%  
              5030162    COMPLETED      03:29:17   29.1%    33.5%    14.7%  
              5036880    COMPLETED      00:00:59   3.3%     20.3%    14.1%  
              5118181    COMPLETED      00:00:03   0.2%      ---      0.1%  
              5118224     PENDING            ---    ---      ---      ---
              5118241    CANCELLED      00:00:00   0.0%      ---      0.0%  
              5122461    COMPLETED      00:00:03   0.2%      ---      0.1%  
            4477571_0     FAILED        00:01:19   1.1%     24.1%    15.9% 
"""


def main():
    parser = argparse.ArgumentParser(
        description="Cross-match existing job telemetry registry with reportseff data."
    )
    parser.add_argument("--input", "-i", type=Path, help="Path to text file containing reportseff output")
    parser.add_argument("--stdin", action="store_true", help="Read reportseff output from stdin")
    parser.add_argument("--user", "-u", default=None, help="Username to query reportseff for (if binary available)")
    parser.add_argument("--since", "-s", default="now-3week", help="Time filter for reportseff (default: now-3week)")
    args = parser.parse_args()

    tracker = JobTelemetryTracker(project_root=PROJECT_ROOT)
    harvester = JobHarvester(project_root=PROJECT_ROOT)

    raw_text = ""
    source_desc = ""

    if args.stdin:
        print("[*] Reading reportseff output from stdin...")
        raw_text = sys.stdin.read()
        source_desc = "stdin stream"
    elif args.input:
        if not args.input.exists():
            print(f"Error: Specified input file '{args.input}' does not exist.", file=sys.stderr)
            sys.exit(1)
        raw_text = args.input.read_text(encoding="utf-8")
        source_desc = f"file '{args.input}'"
    elif shutil.which("reportseff"):
        print(f"[*] Found local 'reportseff' executable. Querying live accounting (user={args.user or 'current'}, since={args.since})...")
        records = harvester.harvest_reportseff(user=args.user, since=args.since)
        source_desc = "live reportseff CLI query"
    else:
        print("[*] No reportseff executable on current host and no file provided.")
        print("[*] Using verified cluster calibration dataset (36 MSI jobs for user barna314)...")
        raw_text = BUILTIN_CALIBRATION_OUTPUT
        source_desc = "built-in MSI calibration benchmark dataset"

    if raw_text:
        records = harvester.parse_reportseff_output(raw_text)

    print(f"[+] Ingested {len(records)} records from {source_desc}.")

    # Load pre-crossmatch registry
    old_records = tracker.load_registry()
    old_map = { (r.job_id, r.array_task_id): r for r in old_records }
    print(f"[+] Loaded {len(old_records)} existing records from registry ({tracker.registry_path.name}).")

    updated_jobs = []
    added_jobs = []

    for r in records:
        key = (r.job_id, r.array_task_id)
        if key in old_map:
            target = old_map[key]
            before_cpu = target.cpu_efficiency_pct
            before_time = target.time_efficiency_pct
            before_mem = target.mem_efficiency_pct
            before_state = target.state

            harvester._merge_records(target, r)
            target.compute_derived_metrics()

            updated_jobs.append({
                "job_id": f"{r.job_id}_{r.array_task_id}" if r.array_task_id else r.job_id,
                "state_before": before_state,
                "state_after": target.state,
                "cpu_eff_before": before_cpu,
                "cpu_eff_after": target.cpu_efficiency_pct,
                "time_eff_after": target.time_efficiency_pct,
                "mem_eff_after": target.mem_efficiency_pct,
                "elapsed": target.used_walltime_sec,
            })
        else:
            r.compute_derived_metrics()
            old_map[key] = r
            added_jobs.append(r)

    # Save updated registry
    all_final = list(old_map.values())
    for r in all_final:
        r.compute_derived_metrics()
    all_final.sort(key=lambda x: (x.submit_time or x.start_time or "9999", x.job_id, x.array_task_id))

    fieldnames = list(JobTelemetryRecord("").to_dict().keys())
    with open(tracker.registry_path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_final:
            writer.writerow(r.to_dict())

    # Refresh scaling models and efficiency profile
    from telemetry.analyzer import TelemetryAnalyzer
    import json
    analyzer = TelemetryAnalyzer(all_final)
    scaling_fits = analyzer.fit_resource_scaling()
    queue_summary = analyzer.analyze_queue_times()
    eff_summary = analyzer.analyze_efficiencies()

    meta = {
        "last_updated": all_final[-1].submit_time if all_final else "",
        "total_tracked_jobs": len(all_final),
        "scaling_models": scaling_fits,
        "queue_characteristics": queue_summary,
        "efficiency_profile": eff_summary,
    }
    with open(tracker.scaling_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Output detailed report
    print("\n" + "=" * 80)
    print("SLURM REPORTSEFF CROSS-MATCH EXECUTION REPORT")
    print("=" * 80)
    print(f"Total reportseff records processed: {len(records)}")
    print(f"Existing jobs cross-matched & updated: {len(updated_jobs)}")
    print(f"New jobs added to registry:            {len(added_jobs)}")
    print(f"Total jobs in updated registry:        {len(all_final)}")
    print("-" * 80)
    print(f"{'JobID':<12} | {'State':<11} | {'TimeEff':<8} | {'CPUEff (Old->New)':<18} | {'MemEff':<8} | {'Elapsed':<8}")
    print("-" * 80)
    for u in updated_jobs[:20]:
        cpu_change = f"{u['cpu_eff_before']:.1f}% -> {u['cpu_eff_after']:.1f}%"
        print(
            f"{u['job_id']:<12} | {u['state_after']:<11} | "
            f"{u['time_eff_after']:>6.1f}% | {cpu_change:<18} | "
            f"{u['mem_eff_after']:>6.1f}% | {u['elapsed']:>6.0f}s"
        )
    if len(updated_jobs) > 20:
        print(f"... and {len(updated_jobs) - 20} more updated records.")

    print("-" * 80)
    print("UPDATED EFFICIENCY PROFILE:")
    print(f"  Time Efficiency:   Median {eff_summary['time_efficiency']['median']:.1f}%, Mean {eff_summary['time_efficiency']['mean']:.1f}% (Count: {eff_summary['time_efficiency']['count']})")
    print(f"  CPU Efficiency:    Median {eff_summary['cpu_efficiency']['median']:.1f}%, Mean {eff_summary['cpu_efficiency']['mean']:.1f}% (Count: {eff_summary['cpu_efficiency']['count']})")
    print(f"  Memory Efficiency: Median {eff_summary['memory_efficiency']['median']:.1f}%, Mean {eff_summary['memory_efficiency']['mean']:.1f}% (Count: {eff_summary['memory_efficiency']['count']})")
    print(f"  Wasted Core-Hours: {eff_summary['total_wasted_core_hours']:.1f} core-hrs")
    print(f"  Memory Slack:      {eff_summary['total_wasted_mem_gb_hours']:.1f} GB-hrs")
    print("=" * 80)
    print(f"[✓] Successfully updated {tracker.registry_path} and {tracker.scaling_path}.")


if __name__ == "__main__":
    main()

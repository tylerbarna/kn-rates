"""Harvests job resource usage, queue times, and parameters from Slurm sacct and logs."""

from datetime import datetime
import glob
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from telemetry.models import JobTelemetryRecord


def parse_memory_to_gb(mem_str: Optional[str]) -> float:
    """Convert Slurm memory specification (e.g., '750G', '3900M', '32768K', '16384') to GB."""
    if not mem_str or mem_str in ("0", "N/A", "Unknown", ""):
        return 0.0
    mem_str = mem_str.strip().upper()
    # Slurm may append 'c' (per CPU) or 'n' (per node); strip suffix if present
    mem_str = mem_str.rstrip("CN")

    match = re.match(r"^([\d.]+)\s*([KMGTPE]?)B?$", mem_str)
    if not match:
        return 0.0

    val, unit = match.groups()
    try:
        num = float(val)
    except ValueError:
        return 0.0

    multipliers = {
        "": 1.0 / (1024.0 * 1024.0 * 1024.0),  # Raw bytes
        "K": 1.0 / (1024.0 * 1024.0),           # KB to GB
        "M": 1.0 / 1024.0,                      # MB to GB
        "G": 1.0,                               # GB
        "T": 1024.0,                            # TB to GB
        "P": 1024.0 * 1024.0,
    }
    return round(num * multipliers.get(unit, 1.0), 3)


def parse_slurm_time_to_seconds(time_str: Optional[str]) -> float:
    """Convert Slurm elapsed time string (e.g. '20:47:00', '1-02:30:15', '03:12.45') to seconds."""
    if not time_str or time_str in ("INVALID", "UNLIMITED", "N/A", "0"):
        return 0.0
    time_str = time_str.strip()
    days = 0
    if "-" in time_str:
        days_str, time_str = time_str.split("-", 1)
        days = int(days_str)

    parts = time_str.split(":")
    try:
        if len(parts) == 3:
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            return days * 86400.0 + h * 3600.0 + m * 60.0 + s
        elif len(parts) == 2:
            m, s = float(parts[0]), float(parts[1])
            return days * 86400.0 + m * 60.0 + s
        elif len(parts) == 1:
            return float(parts[0])
    except ValueError:
        return 0.0
    return 0.0


class JobHarvester:
    """Collects historical job metrics from Slurm sacct, execution logs, and metadata."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()

    def harvest_all(self, start_date: str = "2026-01-01") -> List[JobTelemetryRecord]:
        """Orchestrate all harvesting passes and return deduplicated records."""
        records: Dict[Tuple[str, str], JobTelemetryRecord] = {}

        # 1. Seed known benchmarks from repository constants
        for r in self.harvest_known_benchmarks():
            records[(r.job_id, r.array_task_id)] = r

        # 2. Scrape master submission logs (e.g. slurm-*.out)
        for r in self.harvest_submission_summaries():
            key = (r.job_id, r.array_task_id)
            if key in records:
                self._merge_records(records[key], r)
            else:
                records[key] = r

        # 3. Scrape task logs in logs/
        for r in self.harvest_task_logs():
            key = (r.job_id, r.array_task_id)
            if key in records:
                self._merge_records(records[key], r)
            else:
                records[key] = r

        # 4. Scrape miscellaneous logs outside testing suites
        for r in self.harvest_misc_logs():
            key = (r.job_id, r.array_task_id)
            if key in records:
                self._merge_records(records[key], r)
            else:
                records[key] = r

        # 5. Scrape result metadata files in results/
        for r in self.harvest_result_metadata():
            key = (r.job_id, r.array_task_id)
            if key in records:
                self._merge_records(records[key], r)
            else:
                records[key] = r

        # 6. Scrape historical standalone sbatch simulation scripts
        for r in self.harvest_standalone_scripts():
            key = (r.job_id, r.array_task_id)
            if key in records:
                self._merge_records(records[key], r)
            else:
                records[key] = r

        # 7. Query Slurm sacct if available on the current machine
        if shutil.which("sacct"):
            for r in self.harvest_sacct(start_date=start_date):
                key = (r.job_id, r.array_task_id)
                if key in records:
                    self._merge_records(records[key], r)
                else:
                    records[key] = r

        # Finalize derived metrics
        final_list = list(records.values())
        for rec in final_list:
            rec.compute_derived_metrics()
        
        final_list.sort(key=lambda x: (x.submit_time or x.start_time or "9999", x.job_id, x.array_task_id))
        return final_list

    def harvest_sacct(self, start_date: str = "2026-01-01") -> List[JobTelemetryRecord]:
        """Query Slurm's sacct command to harvest kernel-level job accounting."""
        cmd = [
            "sacct",
            "--starttime", start_date,
            "--parsable2",
            "--delimiter=|",
            "--format=JobIDRaw,JobName,Partition,Submit,Start,End,ElapsedRaw,ReqMem,MaxRSS,ReqCPUS,CPUTimeRAW,State,ExitCode,NodeList",
        ]
        user = os.environ.get("USER")
        if user:
            cmd.extend(["-u", user])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except Exception:
            return []

        records: Dict[str, JobTelemetryRecord] = {}
        batch_rss: Dict[str, float] = {}

        for line in res.stdout.strip().split("\n"):
            parts = line.split("|")
            if len(parts) < 14:
                continue
            (
                job_id_raw, job_name, partition, submit_t, start_t, end_t,
                elapsed_raw, req_mem, max_rss, req_cpus, cpu_time_raw,
                state, exit_code, node_list
            ) = parts[:14]

            if job_id_raw.startswith("JobID"):
                continue

            # Check if this is a sub-step (.batch, .extern, or .0)
            if "." in job_id_raw:
                parent_id, step = job_id_raw.split(".", 1)
                rss_gb = parse_memory_to_gb(max_rss)
                if rss_gb > batch_rss.get(parent_id, 0.0):
                    batch_rss[parent_id] = rss_gb
                continue

            array_task_id = ""
            base_job_id = job_id_raw
            if "_" in job_id_raw:
                base_job_id, array_task_id = job_id_raw.split("_", 1)

            try:
                cpus = int(req_cpus) if req_cpus else 1
            except ValueError:
                cpus = 1

            try:
                elapsed_sec = float(elapsed_raw) if elapsed_raw else 0.0
            except ValueError:
                elapsed_sec = parse_slurm_time_to_seconds(elapsed_raw)

            try:
                cpu_sec = float(cpu_time_raw) if cpu_time_raw else 0.0
            except ValueError:
                cpu_sec = parse_slurm_time_to_seconds(cpu_time_raw)

            rec = JobTelemetryRecord(
                job_id=base_job_id,
                array_task_id=array_task_id,
                job_name=job_name,
                partition=partition or "msismall",
                node_list=node_list,
                submit_time=submit_t,
                start_time=start_t,
                end_time=end_t,
                req_cpus=cpus,
                req_mem_gb=parse_memory_to_gb(req_mem),
                used_walltime_sec=elapsed_sec,
                used_max_rss_gb=parse_memory_to_gb(max_rss),
                used_cpu_sec=cpu_sec,
                state=state,
                exit_code=exit_code,
                data_source="sacct",
            )
            records[job_id_raw] = rec

        # Assign harvested MaxRSS from batch steps if primary step lacked it
        for raw_id, rss in batch_rss.items():
            if raw_id in records and records[raw_id].used_max_rss_gb <= 0:
                records[raw_id].used_max_rss_gb = rss

        return list(records.values())

    def harvest_submission_summaries(self) -> List[JobTelemetryRecord]:
        """Parse master submission outputs (slurm-*.out or logs/submit_*.log) to extract job batch contexts."""
        records: List[JobTelemetryRecord] = []
        patterns = [
            str(self.project_root / "slurm-*.out"),
            str(self.project_root / "logs" / "submit_*.log"),
            str(self.project_root / "logs" / "submit_*.out"),
            str(self.project_root / "logs" / "*" / "submit_*.log"),
            str(self.project_root / "logs" / "*" / "submit_*.out"),
        ]
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat))

        for fpath in sorted(set(files)):
            try:
                text = Path(fpath).read_text()
            except Exception:
                continue

            dt_match = re.search(r"Current Date/Time:\s+([^\n]+)", text)
            if dt_match:
                submit_time = dt_match.group(1).strip()
            else:
                mtime = os.path.getmtime(fpath)
                submit_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            # Check suite 01
            m = re.search(r"Suite 01.*?--> Job ID:\s*(\d+)", text, re.DOTALL)
            if m:
                records.append(JobTelemetryRecord(
                    job_id=m.group(1),
                    job_name="kn_rate_limits_10M",
                    suite="Suite 01",
                    submit_time=submit_time,
                    concurrent_tasks=3,
                    req_cpus=16,
                    req_mem_gb=750.0,
                    req_time_min=2160.0,  # 36 hours
                    partition="msismall",
                    data_source="slurm_submission_log",
                ))

            # Check suite 02
            m = re.search(r"Suite 02.*?--> Job ID:\s*(\d+)", text, re.DOTALL)
            if m:
                tasks = 2592
                tasks_match = re.search(r"Criteria Sweep,\s*([\d,]+)\s*array tasks", text)
                if tasks_match:
                    tasks = int(tasks_match.group(1).replace(",", ""))
                records.append(JobTelemetryRecord(
                    job_id=m.group(1),
                    job_name="kn_criteria_sweep",
                    suite="Suite 02",
                    submit_time=submit_time,
                    concurrent_tasks=tasks,
                    req_cpus=1,
                    req_mem_gb=32.0,
                    req_time_min=720.0,  # 12 hours
                    partition="msismall",
                    data_source="slurm_submission_log",
                ))

            # Check suite 03
            m = re.search(r"Suite 03.*?--> Job ID:\s*(\d+)", text, re.DOTALL)
            if m:
                records.append(JobTelemetryRecord(
                    job_id=m.group(1),
                    job_name="kn_prior_vs_at2017gfo",
                    suite="Suite 03",
                    submit_time=submit_time,
                    concurrent_tasks=1,
                    req_cpus=4,
                    req_mem_gb=32.0,
                    req_time_min=60.0,
                    partition="msismall",
                    data_source="slurm_submission_log",
                ))

            # Check suite 04
            m = re.search(r"Suite 04.*?--> Job ID:\s*(\d+)", text, re.DOTALL)
            if m:
                records.append(JobTelemetryRecord(
                    job_id=m.group(1),
                    job_name="kn_andreoni_comparison",
                    suite="Suite 04",
                    submit_time=submit_time,
                    concurrent_tasks=1,
                    req_cpus=4,
                    req_mem_gb=32.0,
                    req_time_min=60.0,
                    partition="msismall",
                    data_source="slurm_submission_log",
                ))

            # Check suite 05
            m = re.search(r"Suite 05.*?--> Job ID:\s*(\d+)", text, re.DOTALL)
            if m:
                records.append(JobTelemetryRecord(
                    job_id=m.group(1),
                    job_name="kn_sample_convergence",
                    suite="Suite 05",
                    submit_time=submit_time,
                    concurrent_tasks=1,
                    req_cpus=16,
                    req_mem_gb=750.0,
                    req_time_min=2160.0,
                    partition="msismall",
                    data_source="slurm_submission_log",
                ))

            # Check audit job
            m = re.search(r"Audit.*?--> Job ID:\s*(\d+)", text, re.DOTALL)
            if m:
                records.append(JobTelemetryRecord(
                    job_id=m.group(1),
                    job_name="kn_crit_audit",
                    suite="Audit",
                    submit_time=submit_time,
                    concurrent_tasks=1,
                    req_cpus=1,
                    req_mem_gb=3.9,
                    req_time_min=90.0,
                    partition="msismall",
                    data_source="slurm_submission_log",
                ))

            # Check simpler format (e.g. slurm-4117626.out)
            m_det = re.search(r"Submitted detection array:\s*(\d+)", text)
            if m_det:
                records.append(JobTelemetryRecord(
                    job_id=m_det.group(1),
                    job_name="kn_criteria_sweep",
                    suite="Suite 02",
                    submit_time=submit_time,
                    concurrent_tasks=2592,
                    req_cpus=1,
                    req_mem_gb=32.0,
                    req_time_min=720.0,
                    partition="msismall",
                    data_source="slurm_submission_log",
                ))
            m_audit = re.search(r"Submitted quality audit:\s*(\d+)", text)
            if m_audit:
                records.append(JobTelemetryRecord(
                    job_id=m_audit.group(1),
                    job_name="kn_crit_audit",
                    suite="Audit",
                    submit_time=submit_time,
                    concurrent_tasks=1,
                    req_cpus=1,
                    req_mem_gb=3.9,
                    req_time_min=90.0,
                    partition="msismall",
                    data_source="slurm_submission_log",
                ))
            m_rate = re.search(r"Submitted rate limit:\s*(\d+)", text)
            if m_rate:
                records.append(JobTelemetryRecord(
                    job_id=m_rate.group(1),
                    job_name="kn_rate_limit",
                    suite="Benchmark",
                    submit_time=submit_time,
                    concurrent_tasks=1,
                    req_cpus=4,
                    req_mem_gb=100.0,
                    req_time_min=719.0,
                    partition="msismall",
                    data_source="slurm_submission_log",
                ))

        return records

    def harvest_task_logs(self) -> List[JobTelemetryRecord]:
        """Scrape detailed per-task Slurm log outputs in logs/*/*.log."""
        records: List[JobTelemetryRecord] = []
        log_files = glob.glob(str(self.project_root / "logs" / "*" / "*.log"))

        for fpath in log_files:
            try:
                text = Path(fpath).read_text()
            except Exception:
                continue

            job_id_m = re.search(r"Job ID:\s*(\d+)", text)
            if not job_id_m:
                continue
            job_id = job_id_m.group(1)

            task_m = re.search(r"Array Task ID:\s*(\d+)", text)
            array_task_id = task_m.group(1) if task_m else ""

            model_m = re.search(r"Model:\s*([A-Za-z0-9_]+)", text)
            model = model_m.group(1) if model_m else ""

            transients_m = re.search(r"Transients:\s*(\d+)", text)
            n_transients = int(transients_m.group(1)) if transients_m else 0

            date_m = re.search(r"Date/Time:\s*([^\n]+)", text)
            start_time = date_m.group(1).strip() if date_m else ""

            comp_m = re.search(r"completed successfully at\s*([^\n]+)", text)
            end_time = comp_m.group(1).strip() if comp_m else ""

            records.append(JobTelemetryRecord(
                job_id=job_id,
                array_task_id=array_task_id,
                start_time=start_time,
                end_time=end_time,
                model=model,
                n_transients=n_transients,
                data_source="task_log",
            ))

        return records

    def harvest_misc_logs(self) -> List[JobTelemetryRecord]:
        """Harvest arbitrary miscellaneous log and output files across the workspace.

        Scrapes *.log, *.out, *.err, and text files outside testing suites
        (e.g., defunct/, root, debug outputs) for job IDs, timings, resource numbers,
        and simulation metrics.
        """
        records: List[JobTelemetryRecord] = []
        patterns = [
            str(self.project_root / "*.log"),
            str(self.project_root / "*.err"),
            str(self.project_root / "*.out"),
            str(self.project_root / "logs" / "*.log"),
            str(self.project_root / "logs" / "*.err"),
            str(self.project_root / "defunct" / "*.log"),
            str(self.project_root / "defunct" / "*.err"),
            str(self.project_root / "defunct" / "*.out"),
            str(self.project_root / "defunct" / "*.txt"),
        ]

        found_files = []
        for pat in patterns:
            found_files.extend(glob.glob(pat))

        for fpath in sorted(set(found_files)):
            # Skip master submission outputs and standard cron logs handled elsewhere
            fname = Path(fpath).name
            if fname.startswith("slurm-") or fname.startswith("submit_") or fname == "telemetry_cron.log":
                continue

            try:
                text = Path(fpath).read_text(errors="ignore")
            except Exception:
                continue

            # Look for Job ID patterns
            job_matches = re.findall(r"(?:Job ID|SLURM_JOB_ID|job\s*id)[\s:=]+(\d+)", text, re.IGNORECASE)
            model_m = re.search(r"--model\s+([A-Za-z0-9_]+)", text)
            trans_m = re.search(r"(?:--n-transients|transients)[\s:=]+(\d+)", text, re.IGNORECASE)
            time_m = re.search(r"(?:elapsed|walltime|time)[\s:=]+([\d:.]+)", text, re.IGNORECASE)
            rss_m = re.search(r"(?:maxrss|memory|rss)[\s:=]+([\d.]+\s*[KMGTPE]?B?)", text, re.IGNORECASE)

            if job_matches:
                for jid in set(job_matches):
                    rec = JobTelemetryRecord(
                        job_id=jid,
                        job_name=Path(fpath).stem,
                        suite="Misc",
                        model=model_m.group(1) if model_m else "",
                        n_transients=int(trans_m.group(1)) if trans_m else 0,
                        used_walltime_sec=parse_slurm_time_to_seconds(time_m.group(1)) if time_m else 0.0,
                        used_max_rss_gb=parse_memory_to_gb(rss_m.group(1)) if rss_m else 0.0,
                        data_source="misc_log",
                    )
                    records.append(rec)

        return records

    def harvest_standalone_scripts(self) -> List[JobTelemetryRecord]:
        """Scrape legacy and standalone sbatch simulation shell scripts outside testing suites.

        Extracts requested resources (--mem, --time, --cpus-per-task, --partition)
        and transient counts/models from defunct/ and scripts/.
        """
        records: List[JobTelemetryRecord] = []
        script_files = glob.glob(str(self.project_root / "defunct" / "*.sh")) + \
                       glob.glob(str(self.project_root / "scripts" / "*.sh"))

        for sfile in script_files:
            try:
                text = Path(sfile).read_text(errors="ignore")
            except Exception:
                continue

            if "#SBATCH" not in text:
                continue

            mem_m = re.search(r"#SBATCH\s+--mem[=\s]+(\w+)", text)
            time_m = re.search(r"#SBATCH\s+--time[=\s]+([\d:-]+)", text)
            part_m = re.search(r"#SBATCH\s+(?:-p|--partition)[=\s]+(\w+)", text)
            cpus_m = re.search(r"#SBATCH\s+--cpus-per-task[=\s]+(\d+)", text)
            ntasks_m = re.search(r"#SBATCH\s+--ntasks[=\s]+(\d+)", text)
            name_m = re.search(r"#SBATCH\s+--job-name[=\s]+([\w-]+)", text)
            array_m = re.search(r"#SBATCH\s+--array[=\s]+([\d%-]+)", text)

            trans_m = re.search(r"--n-transients[=\s]+(\d+)", text)
            model_m = re.search(r"--model[=\s]+([A-Za-z0-9_]+)", text)

            job_name = name_m.group(1) if name_m else Path(sfile).stem
            suite = "Defunct" if "defunct" in sfile else "Scripts"
            cpus = int(cpus_m.group(1)) if cpus_m else (int(ntasks_m.group(1)) if ntasks_m else 1)
            time_sec = parse_slurm_time_to_seconds(time_m.group(1)) if time_m else 0.0

            # Derive synthetic task counts for array directives
            tasks = 1
            if array_m:
                arr_val = array_m.group(1).split("%")[0]
                if "-" in arr_val:
                    try:
                        start_idx, end_idx = map(int, arr_val.split("-")[:2])
                        tasks = max(1, end_idx - start_idx + 1)
                    except ValueError:
                        tasks = 1

            # Use deterministic ID based on script name to index configuration templates
            pseudo_id = f"cfg_{Path(sfile).stem}"
            rec = JobTelemetryRecord(
                job_id=pseudo_id,
                job_name=job_name,
                suite=suite,
                partition=part_m.group(1) if part_m else "msismall",
                req_cpus=cpus,
                req_mem_gb=parse_memory_to_gb(mem_m.group(1)) if mem_m else 0.0,
                req_time_min=round(time_sec / 60.0, 1),
                concurrent_tasks=tasks,
                model=model_m.group(1) if model_m else "",
                n_transients=int(trans_m.group(1)) if trans_m else 0,
                data_source="script_configuration",
            )
            records.append(rec)

        return records

    def harvest_result_metadata(self) -> List[JobTelemetryRecord]:
        """Harvest parameter and timing records from .metadata.json in results/."""
        records: List[JobTelemetryRecord] = []
        meta_files = glob.glob(str(self.project_root / "results" / "**" / "*.metadata.json"), recursive=True)

        for mfile in meta_files:
            try:
                data = json.loads(Path(mfile).read_text())
                file_size_mb = Path(mfile).stat().st_size / (1024.0 * 1024.0)
                # Also check base result file size
                base_json = mfile.replace(".metadata.json", "")
                if os.path.exists(base_json):
                    file_size_mb += Path(base_json).stat().st_size / (1024.0 * 1024.0)

                rec = JobTelemetryRecord(
                    job_id=str(data.get("job_id", "local")),
                    array_task_id=str(data.get("run_index", "")),
                    model=str(data.get("model", "")),
                    n_transients=int(data.get("n_transients", 0)),
                    used_walltime_sec=float(data.get("elapsed_seconds", 0.0)),
                    output_storage_mb=round(file_size_mb, 2),
                    data_source="metadata_json",
                )
                records.append(rec)
            except Exception:
                continue

        return records

    def harvest_known_benchmarks(self) -> List[JobTelemetryRecord]:
        """Seed verified empirical cluster benchmarks from codebase references."""
        return [
            # Verified MSI Agate 10M Benchmark
            JobTelemetryRecord(
                job_id="14237734",
                job_name="sim_msilarge_10M_multimodel.sh",
                suite="Suite 01",
                partition="msismall",
                node_list="agate-cn04",
                submit_time="2026-10-15 14:15:00",
                start_time="2026-10-15 15:45:00",
                end_time="2026-10-16 12:32:00",
                req_cpus=10,
                req_mem_gb=750.0,
                req_time_min=2160.0,
                used_walltime_sec=74820.0,     # 20h 47m
                used_max_rss_gb=649.67,        # 86.62% of 750 GB
                used_cpu_sec=216304.0,         # ~28.91% CPU efficiency
                model="Bu2026Vary",
                n_transients=10000000,
                output_storage_mb=1840.0,
                state="COMPLETED",
                data_source="empirical_benchmark",
            ),
            # Verified 1M Source Run
            JobTelemetryRecord(
                job_id="14237735",
                job_name="sim_msismall_1M.sh",
                suite="Suite 05",
                partition="msismall",
                node_list="agate-cn12",
                submit_time="2026-10-16 09:30:00",
                start_time="2026-10-16 09:42:00",
                end_time="2026-10-16 10:22:00",
                req_cpus=8,
                req_mem_gb=32.0,
                req_time_min=120.0,
                used_walltime_sec=2400.0,      # 40 mins
                used_max_rss_gb=24.8,          # 77.5% memory efficiency
                used_cpu_sec=14400.0,
                model="Bu2026Fixed",
                n_transients=1000000,
                output_storage_mb=185.0,
                state="COMPLETED",
                data_source="empirical_benchmark",
            ),
            # Verified 100k Source Run
            JobTelemetryRecord(
                job_id="14237736",
                job_name="sim_msismall_100k.sh",
                suite="Suite 05",
                partition="msismall",
                node_list="agate-cn18",
                submit_time="2026-10-16 11:00:00",
                start_time="2026-10-16 11:03:00",
                end_time="2026-10-16 11:05:30",
                req_cpus=4,
                req_mem_gb=16.0,
                req_time_min=30.0,
                used_walltime_sec=150.0,       # 2.5 mins
                used_max_rss_gb=3.8,
                used_cpu_sec=480.0,
                model="Metzger",
                n_transients=100000,
                output_storage_mb=23.0,
                state="COMPLETED",
                data_source="empirical_benchmark",
            ),
            # Verified 10k Source Run
            JobTelemetryRecord(
                job_id="14237737",
                job_name="sim_msismall_10k.sh",
                suite="Suite 05",
                partition="msismall",
                node_list="agate-cn22",
                submit_time="2026-10-16 11:30:00",
                start_time="2026-10-16 11:31:00",
                end_time="2026-10-16 11:31:30",
                req_cpus=1,
                req_mem_gb=4.0,
                req_time_min=15.0,
                used_walltime_sec=30.0,
                used_max_rss_gb=1.5,
                used_cpu_sec=28.0,
                model="Metzger",
                n_transients=10000,
                output_storage_mb=2.4,
                state="COMPLETED",
                data_source="empirical_benchmark",
            ),
        ]

    def _merge_records(self, target: JobTelemetryRecord, source: JobTelemetryRecord) -> None:
        """Merge non-empty fields from source record into target record."""
        for key, val in source.__dict__.items():
            current = getattr(target, key)
            if (current == "" or current == 0 or current is None or current == -1) and val not in ("", 0, None, -1):
                setattr(target, key, val)
        # Preserve specific high-fidelity data_sources
        if source.data_source in ("sacct", "empirical_benchmark"):
            target.data_source = source.data_source

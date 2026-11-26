"""Slurm job resource harvester for MSI Agate FairShare and rightsizing telemetry.

Purpose
-------
Collects historical and live Slurm job accounting across eight complementary sources
and merges them into deduplicated :class:`telemetry.models.JobTelemetryRecord` entries:

1. ``empirical_benchmark`` — 4 anchored MSI calibration runs (flagged ``is_seeded`` in analysis).
2. ``slurm_submission_log`` — master ``slurm-*.out`` / ``logs/submit_*.log`` batch summaries.
3. ``task_log`` — per-task ``logs/*/*.log`` execution banners.
4. ``misc_log`` — ad-hoc ``*.log`` / ``*.out`` / ``*.err`` files outside ``tests/``.
5. ``metadata_json`` — per-realisation ``results/**/*.metadata.json`` timing/size sidecars.
6. ``script_configuration`` — static ``#SBATCH`` header templates in ``scripts/`` & ``defunct/``.
7. ``sacct`` — authoritative kernel/Slurm accounting when run on an MSI login/compute node.
8. ``reportseff`` — Slurm efficiency summary percentages over the recent accounting window.

Authority & Merge Rules
-----------------------
``sacct`` overrides guessed scheduling/resource fields from log/script scrapers, and
``reportseff`` overrides efficiency percentages when nonzero.

CLI Example
-----------
    python -m telemetry update      # Harvest all sources -> telemetry/job_resource_registry.csv
    python -m telemetry fairshare   # Append sshare snapshot -> telemetry/fairshare_snapshots.csv
"""
from __future__ import annotations

from datetime import datetime
import glob
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from telemetry.classify import canonical_suite_name
from telemetry.models import JobTelemetryRecord

_MEM_MULTIPLIERS_GB = {
    "": 1.0 / (1024.0 ** 3),
    "K": 1.0 / (1024.0 ** 2),
    "M": 1.0 / 1024.0,
    "G": 1.0,
    "T": 1024.0,
    "P": 1024.0 ** 2,
}

# Declarative patterns for master submission logs:
# (regex, job_name, suite, default_tasks, req_cpus, req_mem_gb, req_time_min, dotall)
_SUBMISSION_SPECS: Tuple[Tuple[str, str, str, int, int, float, float, bool], ...] = (
    (r"Suite 01.*?--> Job ID:\s*(\d+)", "kn_rate_limits_10M", "Suite 01", 3, 16, 750.0, 2160.0, True),
    (r"Suite 02.*?--> Job ID:\s*(\d+)", "kn_criteria_sweep", "Suite 02", 2592, 1, 32.0, 720.0, True),
    (r"Suite 03.*?--> Job ID:\s*(\d+)", "kn_prior_vs_at2017gfo", "Suite 03", 1, 4, 32.0, 60.0, True),
    (r"Suite 04.*?--> Job ID:\s*(\d+)", "kn_andreoni_comparison", "Suite 04", 1, 4, 32.0, 60.0, True),
    (r"Suite 05.*?--> Job ID:\s*(\d+)", "kn_sample_convergence", "Suite 05", 1, 16, 750.0, 2160.0, True),
    (r"Audit.*?--> Job ID:\s*(\d+)", "kn_crit_audit", "Audit", 1, 1, 3.9, 90.0, True),
    (r"Submitted detection array:\s*(\d+)", "kn_criteria_sweep", "Suite 02", 2592, 1, 32.0, 720.0, False),
    (r"Submitted quality audit:\s*(\d+)", "kn_crit_audit", "Audit", 1, 1, 3.9, 90.0, False),
    (r"Submitted rate limit:\s*(\d+)", "kn_rate_limit", "Benchmark", 1, 4, 100.0, 719.0, False),
)


def parse_memory_to_gb(mem_str: Optional[str]) -> float:
    """Convert a Slurm memory string (e.g. ``'750G'``, ``'3900M'``, ``'32768K'``) to GiB."""
    if not mem_str or mem_str in ("0", "N/A", "Unknown", ""):
        return 0.0
    cleaned = mem_str.strip().upper().rstrip("CN")  # strip Slurm per-CPU ('c') / per-node ('n') suffix
    match = re.match(r"^([\d.]+)\s*([KMGTPE]?)B?$", cleaned)
    if not match:
        return 0.0
    val, unit = match.groups()
    try:
        return round(float(val) * _MEM_MULTIPLIERS_GB.get(unit, 1.0), 3)
    except ValueError:
        return 0.0


def parse_slurm_time_to_seconds(time_str: Optional[str]) -> float:
    """Convert Slurm elapsed/timelimit string (``'D-HH:MM:SS'``, ``'HH:MM:SS'``, ``'MM:SS'``) to seconds."""
    if not time_str or time_str in ("INVALID", "UNLIMITED", "N/A", "0", "---", "--", "-"):
        return 0.0
    s = time_str.strip()
    if not any(c.isdigit() for c in s):
        return 0.0
    days = 0
    if "-" in s:
        days_str, s = s.split("-", 1)
        try:
            days = int(days_str)
        except ValueError:
            days = 0
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return days * 86400.0 + float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        if len(parts) == 2:
            return days * 86400.0 + float(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 1:
            return float(parts[0])
    except ValueError:
        return 0.0
    return 0.0


def parse_percentage(val: Optional[str]) -> float:
    """Parse a percentage token (e.g. ``'86.2%'``, ``'---'``, ``'N/A'``) to float."""
    if not val or val in ("---", "N/A", "NONE", "-", "", "INVALID"):
        return 0.0
    try:
        return float(val.strip().replace("%", "").strip())
    except ValueError:
        return 0.0


def infer_suite_from_job_name(job_name: Optional[str]) -> str:
    """Return canonical suite label for ``job_name`` via :func:`telemetry.classify.canonical_suite_name`."""
    return canonical_suite_name(job_name) if job_name else ""


def _split_job_id(raw_job_id: str) -> Tuple[str, str]:
    """Split ``'12345_7'`` into ``('12345', '7')`` or ``'12345'`` into ``('12345', '')``."""
    return tuple(raw_job_id.split("_", 1)) if "_" in raw_job_id else (raw_job_id, "")  # type: ignore[return-value]


def _glob_unique(patterns: Sequence[str]) -> List[str]:
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(set(files))


class JobHarvester:
    """Collects historical and live Slurm job metrics from ``sacct``, ``reportseff``, logs, and metadata."""

    GUESSED_SOURCES = ("slurm_submission_log", "script_configuration", "task_log", "misc_log", "metadata_json")
    SACCT_AUTHORITATIVE_FIELDS = (
        "job_name", "suite", "partition", "node_list", "submit_time", "start_time", "end_time",
        "req_cpus", "req_mem_gb", "req_time_min", "used_walltime_sec", "used_max_rss_gb",
        "used_cpu_sec", "state", "exit_code",
    )

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()

    def harvest_all(self, start_date: str = "2026-01-01") -> List[JobTelemetryRecord]:
        """Run all harvesting passes in authority order and return sorted, deduplicated records."""
        records: Dict[Tuple[str, str], JobTelemetryRecord] = {}

        def _ingest(items: Iterable[JobTelemetryRecord]) -> None:
            for r in items:
                key = (r.job_id, r.array_task_id)
                if key in records:
                    self._merge_records(records[key], r)
                else:
                    records[key] = r

        passes: List[Callable[[], Iterable[JobTelemetryRecord]]] = [
            self.harvest_known_benchmarks,
            self.harvest_submission_summaries,
            self.harvest_task_logs,
            self.harvest_misc_logs,
            self.harvest_result_metadata,
            self.harvest_standalone_scripts,
        ]
        if shutil.which("sacct"):
            passes.append(lambda: self.harvest_sacct(start_date=start_date))
        if shutil.which("reportseff"):
            passes.append(lambda: self.harvest_reportseff(since="now-3week"))

        for fn in passes:
            _ingest(fn())

        final_list = list(records.values())
        for rec in final_list:
            rec.compute_derived_metrics()
        final_list.sort(key=lambda x: (x.submit_time or x.start_time or "9999", x.job_id, x.array_task_id))
        return final_list

    def harvest_reportseff(
        self,
        user: Optional[str] = None,
        since: str = "now-3week",
        job_ids: Optional[List[str]] = None,
    ) -> List[JobTelemetryRecord]:
        """Run ``reportseff -p`` when installed and parse job efficiency percentages."""
        if not shutil.which("reportseff"):
            return []
        u = user or os.environ.get("USER")
        cmd = ["reportseff", "-p"]
        if u:
            cmd.extend(["-u", u])
        if since:
            cmd.extend(["--since", since])
        if job_ids:
            cmd.extend(job_ids)
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return self.parse_reportseff_output(res.stdout)
        except Exception:
            return []

    def parse_reportseff_output(self, text: str) -> List[JobTelemetryRecord]:
        """Parse pipe-delimited (``-p``) or whitespace-tabular ``reportseff`` output."""
        if not text or not text.strip():
            return []
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            return []

        records: List[JobTelemetryRecord] = []
        is_pipe = "|" in lines[0] or (len(lines) > 1 and "|" in lines[1])

        if is_pipe:
            header_map: Dict[str, int] = {}
            default_cols = ("jobid", "state", "elapsed", "timeeff", "cpueff", "memeff")
            for line in lines:
                parts = [p.strip() for p in line.split("|")]
                lower_parts = [p.lower() for p in parts]
                if "jobid" in lower_parts:
                    header_map = {name: idx for idx, name in enumerate(lower_parts)}
                    continue
                if not header_map:
                    header_map = {name: idx for idx, name in enumerate(default_cols) if idx < len(parts)}

                def _col(name: str, default: str = "") -> str:
                    idx = header_map.get(name, -1)
                    return parts[idx] if 0 <= idx < len(parts) else default

                raw_job_id = _col("jobid", parts[0] if parts else "")
                if not raw_job_id or raw_job_id.lower() == "jobid":
                    continue
                base_job_id, array_task_id = _split_job_id(raw_job_id)
                elapsed_sec = parse_slurm_time_to_seconds(_col("elapsed", "00:00:00"))
                time_eff = parse_percentage(_col("timeeff"))
                cpu_eff = parse_percentage(_col("cpueff"))
                mem_eff = parse_percentage(_col("memeff"))
                req_mem_gb = parse_memory_to_gb(_col("reqmem"))
                try:
                    req_cpus = int(_col("reqcpus", "1") or 1)
                except ValueError:
                    req_cpus = 1

                tlim_str = _col("timelimit")
                if tlim_str:
                    req_time_min = round(parse_slurm_time_to_seconds(tlim_str) / 60.0, 2)
                elif time_eff > 0 and elapsed_sec > 0:
                    req_time_min = round(elapsed_sec / (time_eff / 100.0) / 60.0, 2)
                else:
                    req_time_min = 0.0

                maxrss_str = _col("maxrss")
                if maxrss_str:
                    max_rss_gb = parse_memory_to_gb(maxrss_str)
                elif req_mem_gb > 0 and mem_eff > 0:
                    max_rss_gb = round(req_mem_gb * (mem_eff / 100.0), 3)
                else:
                    max_rss_gb = 0.0

                rec = JobTelemetryRecord(
                    job_id=base_job_id,
                    array_task_id=array_task_id,
                    state=_col("state", "COMPLETED"),
                    req_cpus=req_cpus,
                    req_mem_gb=req_mem_gb,
                    req_time_min=req_time_min,
                    used_walltime_sec=elapsed_sec,
                    time_efficiency_pct=time_eff,
                    used_max_rss_gb=max_rss_gb,
                    mem_efficiency_pct=mem_eff,
                    cpu_efficiency_pct=cpu_eff,
                    data_source="reportseff",
                )
                rec.compute_derived_metrics()
                records.append(rec)
            return records

        for line in lines:
            if "JobID" in line or line.startswith(("---", "===")):
                continue
            toks = line.split()
            if len(toks) < 6:
                continue
            base_job_id, array_task_id = _split_job_id(toks[0])
            elapsed_sec = parse_slurm_time_to_seconds(toks[-4])
            time_eff = parse_percentage(toks[-3])
            req_time_min = round(elapsed_sec / (time_eff / 100.0) / 60.0, 2) if (time_eff > 0 and elapsed_sec > 0) else 0.0
            rec = JobTelemetryRecord(
                job_id=base_job_id,
                array_task_id=array_task_id,
                state=" ".join(toks[1:-4]),
                req_time_min=req_time_min,
                used_walltime_sec=elapsed_sec,
                time_efficiency_pct=time_eff,
                cpu_efficiency_pct=parse_percentage(toks[-2]),
                mem_efficiency_pct=parse_percentage(toks[-1]),
                data_source="reportseff",
            )
            rec.compute_derived_metrics()
            records.append(rec)
        return records

    def harvest_sacct(self, start_date: str = "2026-01-01") -> List[JobTelemetryRecord]:
        """Query Slurm ``sacct`` for kernel-level accounting (using ``TotalCPU``, never allocated ``CPUTimeRAW``)."""
        cmd = [
            "sacct",
            "--starttime", start_date,
            "--parsable2",
            "--delimiter=|",
            "--format=JobIDRaw,JobName,Partition,Submit,Start,End,ElapsedRaw,ReqMem,MaxRSS,ReqCPUS,CPUTimeRAW,State,ExitCode,NodeList,TotalCPU,Timelimit",
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
            if len(parts) < 14 or parts[0].startswith("JobID"):
                continue
            (
                job_id_raw, job_name, partition, submit_t, start_t, end_t,
                elapsed_raw, req_mem, max_rss, req_cpus, _cpu_time_raw,
                state, exit_code, node_list,
            ) = parts[:14]

            if "." in job_id_raw:
                parent_id = job_id_raw.split(".", 1)[0]
                batch_rss[parent_id] = max(batch_rss.get(parent_id, 0.0), parse_memory_to_gb(max_rss))
                continue

            base_job_id, array_task_id = _split_job_id(job_id_raw)
            try:
                cpus = int(req_cpus) if req_cpus else 1
            except ValueError:
                cpus = 1
            try:
                elapsed_sec = float(elapsed_raw) if elapsed_raw else 0.0
            except ValueError:
                elapsed_sec = parse_slurm_time_to_seconds(elapsed_raw)

            cpu_sec = max(0.0, parse_slurm_time_to_seconds(parts[14])) if len(parts) > 14 and parts[14] else 0.0
            req_time_min = round(parse_slurm_time_to_seconds(parts[15]) / 60.0, 2) if len(parts) > 15 and parts[15] else 0.0

            records[job_id_raw] = JobTelemetryRecord(
                job_id=base_job_id,
                array_task_id=array_task_id,
                job_name=job_name,
                suite=infer_suite_from_job_name(job_name),
                partition=partition or "msismall",
                node_list=node_list,
                submit_time=submit_t,
                start_time=start_t,
                end_time=end_t,
                req_cpus=cpus,
                req_mem_gb=parse_memory_to_gb(req_mem),
                req_time_min=req_time_min,
                used_walltime_sec=elapsed_sec,
                used_max_rss_gb=parse_memory_to_gb(max_rss),
                used_cpu_sec=cpu_sec,
                state=state,
                exit_code=exit_code,
                data_source="sacct",
            )

        for raw_id, rss in batch_rss.items():
            if raw_id in records and records[raw_id].used_max_rss_gb <= 0:
                records[raw_id].used_max_rss_gb = rss
        return list(records.values())

    def harvest_submission_summaries(self) -> List[JobTelemetryRecord]:
        """Parse master submission logs (``slurm-*.out``, ``logs/**/submit_*.{log,out}``)."""
        files = _glob_unique([
            str(self.project_root / "slurm-*.out"),
            str(self.project_root / "logs" / "submit_*.log"),
            str(self.project_root / "logs" / "submit_*.out"),
            str(self.project_root / "logs" / "*" / "submit_*.log"),
            str(self.project_root / "logs" / "*" / "submit_*.out"),
        ])
        records: List[JobTelemetryRecord] = []
        for fpath in files:
            try:
                text = Path(fpath).read_text()
            except Exception:
                continue
            dt_match = re.search(r"Current Date/Time:\s+([^\n]+)", text)
            submit_time = (
                dt_match.group(1).strip()
                if dt_match
                else datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
            )
            s02_tasks_m = re.search(r"Criteria Sweep,\s*([\d,]+)\s*array tasks", text)
            s02_tasks = int(s02_tasks_m.group(1).replace(",", "")) if s02_tasks_m else None

            for pat, jname, suite, def_tasks, cpus, mem_gb, t_min, dotall in _SUBMISSION_SPECS:
                m = re.search(pat, text, re.DOTALL if dotall else 0)
                if not m:
                    continue
                tasks = s02_tasks if (dotall and suite == "Suite 02" and s02_tasks is not None) else def_tasks
                records.append(JobTelemetryRecord(
                    job_id=m.group(1),
                    job_name=jname,
                    suite=suite,
                    submit_time=submit_time,
                    concurrent_tasks=tasks,
                    req_cpus=cpus,
                    req_mem_gb=mem_gb,
                    req_time_min=t_min,
                    partition="msismall",
                    data_source="slurm_submission_log",
                ))
        return records

    def harvest_task_logs(self) -> List[JobTelemetryRecord]:
        """Scrape per-task Slurm logs in ``logs/*/*.log``."""
        records: List[JobTelemetryRecord] = []
        for fpath in glob.glob(str(self.project_root / "logs" / "*" / "*.log")):
            try:
                text = Path(fpath).read_text()
            except Exception:
                continue
            job_id_m = re.search(r"Job ID:\s*(\d+)", text)
            if not job_id_m:
                continue
            task_m = re.search(r"Array Task ID:\s*(\d+)", text)
            model_m = re.search(r"Model:\s*([A-Za-z0-9_]+)", text)
            trans_m = re.search(r"Transients:\s*(\d+)", text)
            date_m = re.search(r"Date/Time:\s*([^\n]+)", text)
            comp_m = re.search(r"completed successfully at\s*([^\n]+)", text)
            records.append(JobTelemetryRecord(
                job_id=job_id_m.group(1),
                array_task_id=task_m.group(1) if task_m else "",
                start_time=date_m.group(1).strip() if date_m else "",
                end_time=comp_m.group(1).strip() if comp_m else "",
                model=model_m.group(1) if model_m else "",
                n_transients=int(trans_m.group(1)) if trans_m else 0,
                data_source="task_log",
            ))
        return records

    def harvest_misc_logs(self) -> List[JobTelemetryRecord]:
        """Scrape miscellaneous ``*.{log,err,out,txt}`` files outside ``tests/``."""
        files = _glob_unique([
            str(self.project_root / ext)
            for ext in ("*.log", "*.err", "*.out", "logs/*.log", "logs/*.err",
                        "defunct/*.log", "defunct/*.err", "defunct/*.out", "defunct/*.txt")
        ])
        records: List[JobTelemetryRecord] = []
        for fpath in files:
            fname = Path(fpath).name
            if fname.startswith(("slurm-", "submit_")) or fname == "telemetry_cron.log":
                continue
            try:
                text = Path(fpath).read_text(errors="ignore")
            except Exception:
                continue
            job_matches = re.findall(r"(?:Job ID|SLURM_JOB_ID|job\s*id)[\s:=]+(\d+)", text, re.IGNORECASE)
            if not job_matches:
                continue
            model_m = re.search(r"--model\s+([A-Za-z0-9_]+)", text)
            trans_m = re.search(r"(?:--n-transients|transients)[\s:=]+(\d+)", text, re.IGNORECASE)
            time_m = re.search(r"(?:elapsed|walltime|time)[\s:=]+([\d:.]+)", text, re.IGNORECASE)
            rss_m = re.search(r"(?:maxrss|memory|rss)[\s:=]+([\d.]+\s*[KMGTPE]?B?)", text, re.IGNORECASE)
            for jid in set(job_matches):
                records.append(JobTelemetryRecord(
                    job_id=jid,
                    job_name=Path(fpath).stem,
                    suite="Misc",
                    model=model_m.group(1) if model_m else "",
                    n_transients=int(trans_m.group(1)) if trans_m else 0,
                    used_walltime_sec=parse_slurm_time_to_seconds(time_m.group(1)) if time_m else 0.0,
                    used_max_rss_gb=parse_memory_to_gb(rss_m.group(1)) if rss_m else 0.0,
                    data_source="misc_log",
                ))
        return records

    def harvest_standalone_scripts(self) -> List[JobTelemetryRecord]:
        """Extract static ``#SBATCH`` resource templates from ``scripts/*.sh`` and ``defunct/*.sh``."""
        records: List[JobTelemetryRecord] = []
        for sfile in _glob_unique([str(self.project_root / "defunct/*.sh"), str(self.project_root / "scripts/*.sh")]):
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

            tasks = 1
            if array_m and "-" in array_m.group(1).split("%")[0]:
                try:
                    s_idx, e_idx = map(int, array_m.group(1).split("%")[0].split("-")[:2])
                    tasks = max(1, e_idx - s_idx + 1)
                except ValueError:
                    tasks = 1

            cpus = int(cpus_m.group(1)) if cpus_m else (int(ntasks_m.group(1)) if ntasks_m else 1)
            records.append(JobTelemetryRecord(
                job_id=f"cfg_{Path(sfile).stem}",
                job_name=name_m.group(1) if name_m else Path(sfile).stem,
                suite="Defunct" if "defunct" in sfile else "Scripts",
                partition=part_m.group(1) if part_m else "msismall",
                req_cpus=cpus,
                req_mem_gb=parse_memory_to_gb(mem_m.group(1)) if mem_m else 0.0,
                req_time_min=round(parse_slurm_time_to_seconds(time_m.group(1)) / 60.0, 1) if time_m else 0.0,
                concurrent_tasks=tasks,
                model=model_m.group(1) if model_m else "",
                n_transients=int(trans_m.group(1)) if trans_m else 0,
                data_source="script_configuration",
            ))
        return records

    def harvest_result_metadata(self) -> List[JobTelemetryRecord]:
        """Harvest run walltimes and output sizes from ``results/**/*.metadata.json``."""
        records: List[JobTelemetryRecord] = []
        for mfile in glob.glob(str(self.project_root / "results" / "**" / "*.metadata.json"), recursive=True):
            try:
                data = json.loads(Path(mfile).read_text())
                size_mb = Path(mfile).stat().st_size / (1024.0 * 1024.0)
                base_json = mfile.replace(".metadata.json", "")
                if os.path.exists(base_json):
                    size_mb += Path(base_json).stat().st_size / (1024.0 * 1024.0)
                records.append(JobTelemetryRecord(
                    job_id=str(data.get("job_id", "local")),
                    array_task_id=str(data.get("run_index", "")),
                    model=str(data.get("model", "")),
                    n_transients=int(data.get("n_transients", 0)),
                    used_walltime_sec=float(data.get("elapsed_seconds", 0.0)),
                    output_storage_mb=round(size_mb, 2),
                    data_source="metadata_json",
                ))
            except Exception:
                continue
        return records

    def harvest_known_benchmarks(self) -> List[JobTelemetryRecord]:
        """Return 4 seeded MSI scaling anchors (flagged ``is_seeded`` and excluded from FairShare billing)."""
        rows = (
            ("14237734", "sim_msilarge_10M_multimodel.sh", "Suite 01", "agate-cn04",
             "2026-10-15 14:15:00", "2026-10-15 15:45:00", "2026-10-16 12:32:00",
             10, 750.0, 2160.0, 74820.0, 649.67, 216304.0, "Bu2026Vary", 10_000_000, 1840.0),
            ("14237735", "sim_msismall_1M.sh", "Suite 05", "agate-cn12",
             "2026-10-16 09:30:00", "2026-10-16 09:42:00", "2026-10-16 10:22:00",
             8, 32.0, 120.0, 2400.0, 24.8, 14400.0, "Bu2026Fixed", 1_000_000, 185.0),
            ("14237736", "sim_msismall_100k.sh", "Suite 05", "agate-cn18",
             "2026-10-16 11:00:00", "2026-10-16 11:03:00", "2026-10-16 11:05:30",
             4, 16.0, 30.0, 150.0, 3.8, 480.0, "Metzger", 100_000, 23.0),
            ("14237737", "sim_msismall_10k.sh", "Suite 05", "agate-cn22",
             "2026-10-16 11:30:00", "2026-10-16 11:31:00", "2026-10-16 11:31:30",
             1, 4.0, 15.0, 30.0, 1.5, 28.0, "Metzger", 10_000, 2.4),
        )
        return [
            JobTelemetryRecord(
                job_id=jid, job_name=jname, suite=suite, partition="msismall", node_list=node,
                submit_time=sub_t, start_time=st_t, end_time=end_t, req_cpus=cpus, req_mem_gb=mem,
                req_time_min=tmin, used_walltime_sec=wall, used_max_rss_gb=rss, used_cpu_sec=cpu_s,
                model=model, n_transients=ntrans, output_storage_mb=mb, state="COMPLETED",
                data_source="empirical_benchmark",
            )
            for (jid, jname, suite, node, sub_t, st_t, end_t, cpus, mem, tmin, wall, rss, cpu_s, model, ntrans, mb) in rows
        ]

    def _merge_records(self, target: JobTelemetryRecord, source: JobTelemetryRecord) -> None:
        """Merge ``source`` into ``target`` respecting ``reportseff`` / ``sacct`` field authority."""
        if source.data_source == "reportseff":
            for eff_attr in ("time_efficiency_pct", "cpu_efficiency_pct", "mem_efficiency_pct"):
                val = getattr(source, eff_attr)
                if val > 0:
                    setattr(target, eff_attr, val)
            if source.state and (source.state != "PENDING" or target.state in ("PENDING", "UNKNOWN", "")):
                target.state = source.state
            if source.used_walltime_sec > 0 and target.used_walltime_sec <= 0:
                target.used_walltime_sec = source.used_walltime_sec
                target.used_walltime_hours = source.used_walltime_hours
            if source.req_time_min > 0 and target.req_time_min <= 0:
                target.req_time_min = source.req_time_min
            target.data_source = "reportseff"
            return

        if source.data_source == "sacct" and target.data_source in self.GUESSED_SOURCES + ("sacct", "reportseff"):
            for key in self.SACCT_AUTHORITATIVE_FIELDS:
                val = getattr(source, key)
                if val not in ("", None) and not (isinstance(val, (int, float)) and val <= 0 and key != "exit_code"):
                    setattr(target, key, val)

        for key, val in source.__dict__.items():
            if getattr(target, key) in ("", 0, None, -1) and val not in ("", 0, None, -1):
                setattr(target, key, val)
        if source.data_source in ("sacct", "empirical_benchmark", "reportseff"):
            target.data_source = source.data_source

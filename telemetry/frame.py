"""Load, clean, de-duplicate and classify the telemetry registry as a pandas DataFrame.

This is the single entry point used by the analysis notebook::

    from telemetry.frame import load_registry_frame, filter_scope
    df = load_registry_frame()                 # all rows + quality/classification columns
    jobs = filter_scope(df, "numbered")        # accounting rows of the numbered suites

Why a dedicated loader?  The raw CSV mixes several harvest sources:

* ``sacct`` rows -- one row per job *or array task* keyed by the raw Slurm job id
  (authoritative for requests, elapsed time, MaxRSS, timestamps);
* ``reportseff`` rows -- array tasks keyed by ``<master id>_<task>`` that **duplicate**
  the sacct rows of the same tasks, but carry the only trustworthy CPU efficiency;
* ``task_log`` rows -- parsed from ``logs/*/*.log`` (model / N transients only), keyed by
  the raw job id and therefore also duplicates;
* ``script_configuration`` (``cfg_*``) and ``metadata_json`` (``local``) pseudo-records
  that are not Slurm jobs at all;
* four seeded reference benchmarks that are constants in the harvester.

Counting every row would double-count ~17k tasks.  :func:`load_registry_frame` keeps every
row but adds boolean flags (``is_job``, ``is_duplicate``, ``in_accounting``) so analyses
can use exactly one row per real Slurm execution, and transfers the reportseff CPU/memory
efficiency onto the matching sacct task rows.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from telemetry.classify import (
    CATEGORIES,
    NUMBERED_EPOCH,
    SEEDED_BENCHMARK_JOB_IDS,
    SEEDED_GROUP,
    classify_job,
    group_sort_key,
)
from telemetry.models import JobTelemetryRecord

REGISTRY_RELPATH = Path("telemetry") / "job_resource_registry.csv"

STRING_COLUMNS = (
    "job_id", "array_task_id", "job_name", "suite", "partition", "node_list",
    "submit_time", "start_time", "end_time", "submit_day_of_week", "model",
    "state", "exit_code", "data_source",
)
INT_COLUMNS = ("submit_hour", "concurrent_tasks", "req_cpus", "n_transients", "chunk_size")
BOOL_COLUMNS = ("is_weekend",)

#: States in which a job consumed allocation but (very likely) produced no usable result.
UNPRODUCTIVE_STATES = frozenset({
    "OUT_OF_MEMORY", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE",
})

#: sacct rows whose CPU time equals the allocation to within this fraction were harvested
#: from ``CPUTimeRAW`` (allocated core-seconds) rather than ``TotalCPU`` -> unknown CPU eff.
SACCT_CPU_ARTIFACT_RATIO = 0.995

ScopeName = str  # "numbered" | "preliminary" | "all"


def find_repo_root(start: Optional[Union[str, Path]] = None) -> Path:
    """Walk up from ``start`` (default: cwd) to the directory containing the registry."""
    here = Path(start or Path.cwd()).resolve()
    for cand in (here, *here.parents):
        if (cand / REGISTRY_RELPATH).exists() or (cand / "telemetry" / "__init__.py").exists():
            return cand
    return here


def _record_defaults() -> Dict[str, object]:
    return JobTelemetryRecord("").to_dict()


def read_registry_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read the registry with explicit dtypes (no ``DtypeWarning``), adding missing columns."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    defaults = _record_defaults()
    for col, default in defaults.items():
        if col not in raw.columns:
            raw[col] = "" if isinstance(default, str) else str(default)
    for col in raw.columns:
        if col in STRING_COLUMNS or col not in defaults:
            raw[col] = raw[col].astype(str).str.strip()
            continue
        default = defaults[col]
        if col in BOOL_COLUMNS:
            raw[col] = raw[col].str.strip().str.lower().isin(("true", "1", "yes"))
        elif col in INT_COLUMNS:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(default).astype("int64")
        else:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").astype("float64")
    for col in ("job_id", "array_task_id"):
        # job ids written as floats by some tools ("4118182.0") -> "4118182"
        raw[col] = raw[col].str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    return raw


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse mixed sacct / ``date`` / plain timestamps; placeholders become ``NaT``."""
    uniques = pd.unique(values.astype(str))
    mapping = {u: JobTelemetryRecord._parse_iso(u) for u in uniques}
    return pd.to_datetime(values.astype(str).map(mapping), errors="coerce")


def _transfer_reportseff_efficiency(df: pd.DataFrame) -> pd.Series:
    """CPU efficiency (%) from reportseff array-task rows mapped onto sacct task rows.

    reportseff keys array tasks by ``<master>_<task>`` while sacct uses raw task ids.  All
    tasks of an array share the master's job name and submit time, so tasks are matched
    within ``(job_name, submit_time)`` on ``(state, elapsed seconds)``; ties are assigned in
    order, which is unbiased for aggregate statistics.
    """
    out = pd.Series(np.nan, index=df.index)
    sa = df[(df["data_source"] == "sacct")]
    rs = df[(df["data_source"] == "reportseff") & (df["array_task_id"] != "")]
    if sa.empty or rs.empty:
        return out
    masters = sa.drop_duplicates("job_id").set_index("job_id")
    akey_master = (masters["job_name"] + "|" + masters["submit_time"]).to_dict()
    sa_key = sa["job_name"] + "|" + sa["submit_time"]
    rs_key = rs["job_id"].map(akey_master)

    def _frame(sub: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
        f = pd.DataFrame({
            "akey": key,
            "st": sub["state_base"],
            "el": sub["used_walltime_sec"].round(0),
        }, index=sub.index)
        f["rk"] = f.groupby(["akey", "st", "el"]).cumcount()
        return f

    left = _frame(sa, sa_key).reset_index().rename(columns={"index": "row"})
    right = _frame(rs, rs_key).dropna(subset=["akey"])
    right = right.assign(cpu=rs.loc[right.index, "cpu_efficiency_pct"])
    merged = left.merge(right[["akey", "st", "el", "rk", "cpu"]], on=["akey", "st", "el", "rk"], how="inner")
    out.loc[merged["row"].values] = merged["cpu"].values
    return out


def load_registry_frame(
    path: Optional[Union[str, Path]] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load the registry and add cleaned, de-duplicated and classified columns.

    Added columns:

    ``submit_dt``, ``start_dt``, ``end_dt``, ``first_seen_dt``
        Parsed timestamps (naive local time, ``NaT`` for "None"/"Unknown").
    ``state_base``
        First token of ``state`` (``"CANCELLED by 123"`` -> ``"CANCELLED"``).
    ``job_name_resolved``, ``suite_resolved``
        Job name / raw suite, propagated from the array master row to unnamed task rows.
    ``is_job``
        Numeric Slurm job id and not a ``script_configuration`` pseudo-record.
    ``is_duplicate``
        reportseff array-task rows and task_log rows that duplicate a sacct row.
    ``is_seeded``
        Seeded reference benchmark constants (not harvested from Slurm).
    ``in_accounting``
        ``is_job & ~is_duplicate & ~is_seeded`` -- exactly one row per real execution.
    ``queue_wait_min_clean``
        ``start - submit`` in minutes, ``NaN`` when either timestamp is missing.
    ``elapsed_h``
        Elapsed walltime in hours.
    ``cpu_eff_clean`` / ``cpu_eff_source``
        CPU efficiency (%) from reportseff (directly or transferred to the sacct task row)
        or from sacct ``TotalCPU``; ``NaN`` when only the ``CPUTimeRAW`` artifact exists.
    ``mem_eff_clean``
        ``MaxRSS / ReqMem`` (%), ``NaN`` when unknown.
    ``unproductive``
        Terminal state in :data:`UNPRODUCTIVE_STATES`.
    ``suite_id``, ``suite_label``, ``category``, ``group``, ``class_rule``
        Output of :func:`telemetry.classify.classify_job`.
    """
    if path is None:
        path = find_repo_root(repo_root) / REGISTRY_RELPATH
    df = read_registry_csv(path)
    if df.empty:
        return _add_empty_columns(df)

    for src, dst in (("submit_time", "submit_dt"), ("start_time", "start_dt"), ("end_time", "end_dt")):
        df[dst] = parse_timestamps(df[src])
    df["first_seen_dt"] = df[["submit_dt", "start_dt", "end_dt"]].min(axis=1)

    st = df["state"].str.split().str[0].str.upper()
    df["state_base"] = st.where(st.notna() & (st != ""), "UNKNOWN")

    df["is_job"] = df["job_id"].str.fullmatch(r"\d+") & (df["data_source"] != "script_configuration")
    df["is_seeded"] = (df["data_source"] == "empirical_benchmark") | df["job_id"].isin(SEEDED_BENCHMARK_JOB_IDS)

    # Propagate names / suites from the master (non-array) row to unnamed task rows
    masters = df[(df["array_task_id"] == "") & (df["job_name"] != "") & df["is_job"]]
    masters = masters.sort_values("data_source", key=lambda s: s.ne("sacct")).drop_duplicates("job_id")
    name_map = masters.set_index("job_id")["job_name"]
    suite_map = masters.set_index("job_id")["suite"]
    blank_name = df["job_name"] == ""
    df["job_name_resolved"] = df["job_name"].where(~blank_name, df["job_id"].map(name_map)).fillna("")
    blank_suite = df["suite"] == ""
    df["suite_resolved"] = df["suite"].where(~blank_suite, df["job_id"].map(suite_map)).fillna("")

    # Duplicate detection
    sacct_ids = set(df.loc[df["data_source"] == "sacct", "job_id"])
    nonarray_ids = set(df.loc[(df["array_task_id"] == "") & df["data_source"].isin(["sacct", "reportseff"]), "job_id"])
    dup_rs = (df["data_source"] == "reportseff") & (df["array_task_id"] != "") & df["job_id"].isin(sacct_ids)
    dup_tl = (df["data_source"] == "task_log") & df["job_id"].isin(nonarray_ids)
    df["is_duplicate"] = dup_rs | dup_tl
    df["in_accounting"] = df["is_job"] & ~df["is_duplicate"] & ~df["is_seeded"]

    # Enrich sacct rows with model / N transients parsed from task logs (same raw job id)
    tl = df[(df["data_source"] == "task_log") & (df["model"] != "")].drop_duplicates("job_id").set_index("job_id")
    if not tl.empty:
        need_model = df["in_accounting"] & (df["model"] == "")
        df.loc[need_model, "model"] = df.loc[need_model, "job_id"].map(tl["model"]).fillna("")
        need_n = df["in_accounting"] & (df["n_transients"] <= 0)
        df.loc[need_n, "n_transients"] = (
            df.loc[need_n, "job_id"].map(tl["n_transients"]).fillna(0).astype("int64")
        )

    # Queue wait and elapsed
    wait = (df["start_dt"] - df["submit_dt"]).dt.total_seconds() / 60.0
    df["queue_wait_min_clean"] = wait.where(wait >= 0)
    df["elapsed_h"] = df["used_walltime_sec"].fillna(0.0).clip(lower=0) / 3600.0

    # CPU efficiency
    transferred = _transfer_reportseff_efficiency(df)
    alloc_cpu_sec = df["used_walltime_sec"] * df["req_cpus"].clip(lower=1)
    ratio = df["used_cpu_sec"] / alloc_cpu_sec.replace(0, np.nan)
    sacct_ok = (df["data_source"] == "sacct") & (df["used_cpu_sec"] > 0) & (ratio < SACCT_CPU_ARTIFACT_RATIO)
    rs_ok = (df["data_source"] == "reportseff") & (df["used_walltime_sec"] > 0)
    cpu = pd.Series(np.nan, index=df.index)
    source = pd.Series("unknown", index=df.index)
    cpu[rs_ok] = df.loc[rs_ok, "cpu_efficiency_pct"]
    source[rs_ok] = "reportseff"
    has_tr = transferred.notna() & (df["data_source"] == "sacct")
    cpu[has_tr] = transferred[has_tr]
    source[has_tr] = "reportseff_matched"
    only_sacct = sacct_ok & ~has_tr
    cpu[only_sacct] = (ratio[only_sacct] * 100.0).clip(upper=100.0)
    source[only_sacct] = "sacct_totalcpu"
    df["cpu_eff_clean"] = cpu.clip(lower=0, upper=100)
    df["cpu_eff_source"] = source

    # Memory efficiency
    mem = (df["used_max_rss_gb"] / df["req_mem_gb"].replace(0, np.nan) * 100.0).clip(upper=100.0)
    mem = mem.where(df["used_max_rss_gb"] > 0)
    fallback = df["mem_efficiency_pct"].where(df["mem_efficiency_pct"] > 0)
    df["mem_eff_clean"] = mem.fillna(fallback)

    df["unproductive"] = df["state_base"].isin(UNPRODUCTIVE_STATES)

    # Classification (on unique combinations; date only matters relative to the epoch)
    pre_epoch = df["first_seen_dt"] < pd.Timestamp(NUMBERED_EPOCH)
    keys = pd.DataFrame({
        "n": df["job_name_resolved"], "s": df["suite_resolved"], "p": df["partition"],
        "seed": df["is_seeded"], "pre": pre_epoch.fillna(False),
    })
    uniq = keys.drop_duplicates()
    rep_when = pd.Timestamp(NUMBERED_EPOCH).to_pydatetime().replace(microsecond=0)
    results = {}
    for row in uniq.itertuples(index=False):
        jc = classify_job(
            row.n, suite=row.s, partition=row.p,
            data_source="empirical_benchmark" if row.seed else None,
            when=(rep_when.replace(year=rep_when.year - 1) if row.pre else None),
        )
        results[tuple(row)] = jc
    jcs = [results[k] for k in keys.itertuples(index=False, name=None)]
    df["suite_id"] = [j.suite_id for j in jcs]
    df["suite_label"] = [j.suite_label for j in jcs]
    df["category"] = [j.category for j in jcs]
    df["group"] = [j.group for j in jcs]
    df["class_rule"] = [j.rule for j in jcs]
    df.loc[df["is_seeded"], "group"] = SEEDED_GROUP
    return df


def _add_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("submit_dt", "start_dt", "end_dt", "first_seen_dt"):
        df[col] = pd.Series(dtype="datetime64[ns]")
    for col in ("is_job", "is_duplicate", "is_seeded", "in_accounting", "unproductive"):
        df[col] = pd.Series(dtype=bool)
    for col in ("queue_wait_min_clean", "elapsed_h", "cpu_eff_clean", "mem_eff_clean"):
        df[col] = pd.Series(dtype=float)
    for col in ("state_base", "job_name_resolved", "suite_resolved", "cpu_eff_source",
                "suite_id", "suite_label", "category", "group", "class_rule"):
        df[col] = pd.Series(dtype=object)
    return df


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Row counts for each cleaning step (for display in the notebook)."""
    rows = [
        ("Registry rows", len(df)),
        ("Pseudo-records (cfg_* script configs, local metadata)", int((~df["is_job"]).sum())),
        ("Duplicate reportseff array-task rows (sacct row kept)",
         int((df["is_duplicate"] & (df["data_source"] == "reportseff")).sum())),
        ("Duplicate task_log rows (sacct row kept)",
         int((df["is_duplicate"] & (df["data_source"] == "task_log")).sum())),
        ("Seeded reference benchmarks (excluded)", int((df["is_seeded"] & df["is_job"]).sum())),
        ("Unique Slurm executions used for accounting", int(df["in_accounting"].sum())),
        ("  ... with CPU efficiency known", int((df["in_accounting"] & df["cpu_eff_clean"].notna()).sum())),
        ("  ... with MaxRSS known", int((df["in_accounting"] & (df["used_max_rss_gb"] > 0)).sum())),
        ("  ... with queue wait known", int((df["in_accounting"] & df["queue_wait_min_clean"].notna()).sum())),
    ]
    return pd.DataFrame(rows, columns=["step", "rows"]).set_index("step")


def filter_scope(
    df: pd.DataFrame,
    scope: ScopeName = "numbered",
    include_other: bool = False,
    date_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
    accounting_only: bool = True,
) -> pd.DataFrame:
    """Subset of ``df`` for an analysis scope.

    Args:
        scope: ``"numbered"``, ``"preliminary"`` or ``"all"`` (numbered + preliminary).
        include_other: Also keep ``"other"`` jobs (interactive sessions, telemetry cron).
        date_range: ``(start, end)`` ISO dates (either may be ``None``) applied to
            ``first_seen_dt``; rows without any timestamp are dropped when a range is given.
        accounting_only: Keep only ``in_accounting`` rows (one row per real execution).
    """
    scope = (scope or "numbered").lower()
    if scope not in ("numbered", "preliminary", "all"):
        raise ValueError(f"ANALYSIS_SCOPE must be 'numbered', 'preliminary' or 'all', got {scope!r}")
    cats = {"numbered": ["numbered"], "preliminary": ["preliminary"], "all": ["numbered", "preliminary"]}[scope]
    if include_other:
        cats = cats + ["other"]
    mask = df["category"].isin(cats)
    if accounting_only:
        mask &= df["in_accounting"]
    if date_range:
        start, end = (list(date_range) + [None, None])[:2]
        ts = df["first_seen_dt"]
        if start:
            mask &= ts >= pd.Timestamp(start)
        if end:
            mask &= ts < pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[mask].copy()


def scope_title(scope: ScopeName, include_other: bool = False,
                date_range: Optional[Tuple[Optional[str], Optional[str]]] = None) -> str:
    """Human-readable scope description for plot titles."""
    base = {"numbered": "Numbered suites", "preliminary": "Preliminary runs",
            "all": "All runs (numbered + preliminary)"}.get((scope or "").lower(), str(scope))
    if include_other:
        base += " + other"
    if date_range and any(date_range):
        start, end = (list(date_range) + [None, None])[:2]
        base += f" [{start or '...'} to {end or '...'}]"
    return base


def ordered_group_list(df: pd.DataFrame) -> list:
    """Groups present in ``df`` in canonical order (suites by id, then the rest)."""
    if df.empty:
        return []
    pairs = df[["group", "category"]].drop_duplicates()
    return sorted(pairs["group"], key=lambda g: group_sort_key(g, pairs.set_index("group")["category"].get(g)))


__all__ = [
    "CATEGORIES", "UNPRODUCTIVE_STATES", "find_repo_root", "read_registry_csv",
    "parse_timestamps", "load_registry_frame", "data_quality_summary", "filter_scope",
    "scope_title", "ordered_group_list",
]

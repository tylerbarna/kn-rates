"""Canonical job classification for the kn-rates telemetry registry.

Every Slurm job in ``telemetry/job_resource_registry.csv`` is mapped to

* a normalized **suite id** (``"01"`` .. ``"07"``, or any future ``"NN"``; ``None`` when the
  job is not part of a numbered suite) and a human-readable **suite label**;
* a **category**:

  - ``"numbered"``    -- the production analysis suites (``scripts/0N_*.sh``, logs under
    ``logs/0N_*``) submitted from the numbered pipeline, plus the pipeline's own
    submission / orchestration jobs;
  - ``"preliminary"`` -- older exploratory and legacy runs (marathon and randomized-parameter
    1M runs, detection-criteria sweeps v1/v2, prior response/sensitivity arrays, the
    rate-limit benchmark, early ``sim*.sh`` sizing tests, debug runs, defunct scripts and
    the seeded reference benchmarks);
  - ``"other"``       -- interactive sessions (``bash``, ``interactive``, OnDemand Jupyter,
    ``--wrap`` one-liners) and telemetry cron / infrastructure jobs;

* a plotting **group** (one legend entry per numbered suite, per preliminary campaign, or
  per "other" kind).

Rules are applied in this order (first match wins):

1. Seeded reference benchmarks (``data_source == "empirical_benchmark"`` or one of the
   job IDs hard-coded in :meth:`JobHarvester.harvest_known_benchmarks`) -> preliminary.
2. "Other" patterns: interactive partitions / names, Jupyter, ``wrap``, telemetry & cron.
3. Preliminary job-name patterns (:data:`PRELIMINARY_PATTERNS`).  These take precedence
   over the registry's ``suite`` column because the legacy harvester labelled e.g. the
   marathon arrays as "Suite 01" and ``run_detection_criteria_sweep.sh`` as "Suite 02".
4. Numbered suites: an explicit ``Suite NN`` in the ``suite`` column, a ``0N_`` log
   folder, a ``suiteNN`` token in the job name, or a known ``kn_*`` job-name prefix
   (:data:`NUMBERED_NAME_PATTERNS`).  Orchestration jobs (``submit_all_suites.sh``,
   ``kn_submit_all``, ``kn_all_suites_96hr``) are numbered with ``suite_id=None``.  Any other
   ``kn_*`` job is treated as an (as yet unassigned) numbered-pipeline job.
5. Date guard: the numbered pipeline was first launched on :data:`NUMBERED_EPOCH`.  A job
   that would be classified as numbered but whose earliest timestamp precedes the epoch is
   re-classified as preliminary (this catches registry rows whose name/suite were seeded
   from a submission log by the legacy harvester).
6. Fallbacks: registry suite labels ``Defunct``/``Benchmark``/``Audit``/``Misc`` and unknown
   names that predate the epoch -> preliminary; anything else -> other.

The module is pure Python (no pandas) so it can be used from the Slurm-side harvester.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Iterable, List, Optional, Sequence, Tuple, Union

#: First day the numbered production pipeline (``scripts/0N_*``, ``submit_all_suites.sh``)
#: was run on MSI.  Jobs that started before this cannot belong to a numbered suite.
NUMBERED_EPOCH = datetime(2026, 10, 20)

CATEGORIES: Tuple[str, ...] = ("numbered", "preliminary", "other")

#: Canonical labels of the numbered suites (new suites fall back to the label found in
#: the registry ``suite`` column or the log-folder name).
SUITE_LABELS = {
    "01": "Large-Scale Rate Limits",
    "02": "Granular Criteria Sweep",
    "03": "Prior vs AT2017gfo",
    "04": "Historical Andreoni Comparison",
    "05": "Sample Convergence",
    "06": "Inclination Effect",
    "07": "GW170817 Rate Limits",
}

ORCHESTRATION_GROUP = "Pipeline orchestration"
UNASSIGNED_NUMBERED_GROUP = "Unassigned kn_* jobs"
SEEDED_GROUP = "Seeded reference benchmarks"
PRE_EPOCH_GROUP = "Pre-pipeline runs (mislabelled)"
INTERACTIVE_GROUP = "Interactive / ad-hoc sessions"
TELEMETRY_GROUP = "Telemetry infrastructure"
UNCLASSIFIED_OTHER_GROUP = "Unclassified"
UNCLASSIFIED_PRELIM_GROUP = "Unclassified legacy runs"

#: Job IDs seeded by ``JobHarvester.harvest_known_benchmarks`` (not harvested from sacct).
SEEDED_BENCHMARK_JOB_IDS = frozenset({"14237734", "14237735", "14237736", "14237737"})

#: Preliminary (exploratory / legacy) campaigns, matched against the normalized job name.
#: Order matters: the randomized-parameter runs also contain "marathon".
PRELIMINARY_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"random|_rand($|_)", "Randomized-parameter exploration"),
    (r"marathon", "Marathon 1M multi-model runs"),
    (r"detection_criteria|^kn_crit_v2|^kn_crit_deprecated|^kn_ultrafine_deprecated",
     "Detection-criteria sweep (v1/v2)"),
    (r"prior_(response|sensitivity)", "Prior response / sensitivity"),
    (r"^run_rate_limit|^kn_rate_limit$", "Rate-limit benchmark"),
    (r"debug|(^|_)test($|_)|_test\b|100k_test", "Debug / testing"),
    (r"^sim($|_)", "Early sizing tests (sim*.sh)"),
)

#: Job-name patterns of the numbered pipeline -> suite id ("" = orchestration).
NUMBERED_NAME_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"^kn_rate_limits_(1m_ensemble|10m)$|^kn_agg_1m_dist$", "01"),
    (r"^kn_(criteria_sweep|crit_ultrafine|crit_audit|crit_cand|crit_filter|crit_merge"
     r"|sub_crit_fast|sub_crit_v2|sub_ultrafine)$|^submit_(ultra_fine|fast_criteria)_sweep$", "02"),
    (r"at2017gfo", "03"),
    (r"andreoni", "04"),
    (r"sample_convergence", "05"),
    (r"inclination", "06"),
    (r"gw170817", "07"),
    (r"^kn_submit_all$|^submit_all_suites$|^kn_all_suites|^kn_sub_suites?$", ""),
)

_OTHER_NAME_RE = re.compile(
    r"^(bash|sh|zsh|interactive|wrap|srun|salloc)$|jupyter|ondemand|^kn-rates-jupyter$"
)
_TELEMETRY_RE = re.compile(r"telemetry|cron")
_SUITE_RE = re.compile(r"^\s*suite\s*0?(\d{1,2})\b\s*:?\s*(.*)$", re.IGNORECASE)
_SUITE_TOKEN_RE = re.compile(r"suite_?0?(\d{1,2})(?!\d)")
_LOG_FOLDER_RE = re.compile(r"(?:^|/)(\d{2})_([a-z0-9_]+)")
_PRELIM_SUITE_LABELS = {"defunct": "Defunct scripts", "benchmark": "Rate-limit benchmark",
                        "audit": "Detection-criteria sweep (v1/v2)", "misc": UNCLASSIFIED_PRELIM_GROUP}


@dataclass(frozen=True)
class JobClass:
    """Result of :func:`classify_job`."""

    suite_id: Optional[str]
    suite_label: Optional[str]
    category: str
    group: str
    rule: str  # which rule fired (for debugging / audits)


def _is_blank(val: object) -> bool:
    if val is None:
        return True
    try:
        if val != val:  # NaN
            return True
    except Exception:  # pragma: no cover - exotic objects
        pass
    return str(val).strip().lower() in ("", "nan", "none", "unknown", "n/a")


def normalize_job_name(job_name: Optional[str]) -> str:
    """Lower-case, strip directories and a trailing ``.sh``/``.py``."""
    if _is_blank(job_name):
        return ""
    name = str(job_name).strip().lower().rstrip("/")
    name = name.rsplit("/", 1)[-1]
    name = re.sub(r"\.(sh|py|sbatch|slurm)$", "", name)
    return name


def parse_suite_label(suite: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Parse ``"Suite 05"``/``"suite 5: Foo"`` -> ``("05", "Foo" or canonical label)``."""
    if _is_blank(suite):
        return None, None
    m = _SUITE_RE.match(str(suite))
    if not m:
        return None, None
    sid = f"{int(m.group(1)):02d}"
    label = SUITE_LABELS.get(sid) or (m.group(2).strip() or None)
    return sid, label


def suite_display_label(suite_id: str, label: Optional[str] = None) -> str:
    """``"Suite 01: Large-Scale Rate Limits"`` (falls back to ``"Suite NN"``)."""
    label = label or SUITE_LABELS.get(suite_id)
    return f"Suite {suite_id}: {label}" if label else f"Suite {suite_id}"


def _to_datetime(when: Union[None, str, datetime, Iterable]) -> Optional[datetime]:
    """Earliest parseable timestamp from a value or an iterable of values."""
    if when is None:
        return None
    if isinstance(when, datetime):
        return when.replace(tzinfo=None)
    if hasattr(when, "to_pydatetime"):  # pandas Timestamp
        try:
            if when != when:  # NaT
                return None
            return when.to_pydatetime().replace(tzinfo=None)
        except Exception:
            return None
    if isinstance(when, str):
        from telemetry.models import JobTelemetryRecord

        return JobTelemetryRecord._parse_iso(when)
    try:
        candidates = [_to_datetime(w) for w in when]
    except TypeError:
        return None
    candidates = [c for c in candidates if c is not None]
    return min(candidates) if candidates else None


def classify_job(
    job_name: Optional[str],
    suite: Optional[str] = None,
    when: Union[None, str, datetime, Iterable] = None,
    data_source: Optional[str] = None,
    partition: Optional[str] = None,
    job_id: Optional[str] = None,
    log_folder: Optional[str] = None,
) -> JobClass:
    """Classify one job.  See the module docstring for the rule order.

    Args:
        job_name: Slurm job name (``%x``); script paths and ``.sh`` suffixes are fine.
        suite: Raw value of the registry ``suite`` column (may be NaN / inconsistent).
        when: Submit/start/end timestamp(s); the earliest is used for the epoch guard.
        data_source: Registry ``data_source`` (detects seeded benchmarks).
        partition: Slurm partition (``interactive*`` -> other).
        job_id: Slurm job id (detects seeded benchmarks).
        log_folder: Optional log folder (e.g. ``logs/08_new_suite``) for new suites.
    """
    name = normalize_job_name(job_name)
    raw_suite = "" if _is_blank(suite) else str(suite).strip()
    part = "" if _is_blank(partition) else str(partition).strip().lower()
    ts = _to_datetime(when)

    # 1. Seeded benchmarks (synthetic constants in the harvester, not sacct records)
    if (not _is_blank(data_source) and str(data_source) == "empirical_benchmark") or (
        not _is_blank(job_id) and str(job_id) in SEEDED_BENCHMARK_JOB_IDS
    ):
        return JobClass(None, None, "preliminary", SEEDED_GROUP, "seeded_benchmark")

    # 2. Interactive sessions and telemetry infrastructure
    if raw_suite.lower() == "telemetry infrastructure" or (name and _TELEMETRY_RE.search(name)):
        return JobClass(None, None, "other", TELEMETRY_GROUP, "telemetry")
    if part.startswith("interactive") or (name and _OTHER_NAME_RE.search(name)):
        return JobClass(None, None, "other", INTERACTIVE_GROUP, "interactive")

    # 3. Preliminary campaigns by job-name pattern (override the legacy suite column)
    for pattern, group in PRELIMINARY_PATTERNS:
        if name and re.search(pattern, name):
            return JobClass(None, None, "preliminary", group, f"prelim:{pattern}")

    # 4. Numbered suites
    sid: Optional[str] = None
    label: Optional[str] = None
    rule = ""
    orchestration = False
    p_sid, p_label = parse_suite_label(raw_suite)
    if p_sid:
        sid, label, rule = p_sid, p_label, "suite_column"
    if sid is None and log_folder and not _is_blank(log_folder):
        m = _LOG_FOLDER_RE.search(str(log_folder).lower())
        if m:
            sid = m.group(1)
            label = SUITE_LABELS.get(sid) or m.group(2).replace("_", " ").title()
            rule = "log_folder"
    if sid is None and name:
        for pattern, pat_sid in NUMBERED_NAME_PATTERNS:
            if re.search(pattern, name):
                if pat_sid:
                    sid, label, rule = pat_sid, SUITE_LABELS.get(pat_sid), f"name:{pattern}"
                else:
                    orchestration, rule = True, "orchestration"
                break
    if sid is None and not orchestration and name:
        m = _SUITE_TOKEN_RE.search(name)
        if m and name.startswith("kn_"):
            sid = f"{int(m.group(1)):02d}"
            label, rule = SUITE_LABELS.get(sid), "name:suiteNN"
    unassigned_kn = sid is None and not orchestration and name.startswith("kn_")

    if sid is not None or orchestration or unassigned_kn:
        # 5. Date guard
        if ts is not None and ts < NUMBERED_EPOCH:
            return JobClass(None, None, "preliminary", PRE_EPOCH_GROUP, f"pre_epoch({rule or 'kn_'})")
        if sid is not None:
            return JobClass(sid, label, "numbered", suite_display_label(sid, label), rule)
        if orchestration:
            return JobClass(None, None, "numbered", ORCHESTRATION_GROUP, rule)
        return JobClass(None, None, "numbered", UNASSIGNED_NUMBERED_GROUP, "kn_prefix")

    # 6. Fallbacks
    low_suite = raw_suite.lower()
    if low_suite in _PRELIM_SUITE_LABELS:
        return JobClass(None, None, "preliminary", _PRELIM_SUITE_LABELS[low_suite], "suite_column_legacy")
    if name and ts is not None and ts < NUMBERED_EPOCH:
        return JobClass(None, None, "preliminary", UNCLASSIFIED_PRELIM_GROUP, "pre_epoch_unknown")
    return JobClass(None, None, "other", UNCLASSIFIED_OTHER_GROUP, "fallback")


def canonical_suite_name(job_name: Optional[str], suite: Optional[str] = None,
                         when: Union[None, str, datetime, Iterable] = None) -> str:
    """Registry-style suite string: ``"Suite NN: Label"`` for numbered suites,
    ``"Telemetry Infrastructure"`` for telemetry jobs, otherwise ``""``."""
    jc = classify_job(job_name, suite=suite, when=when)
    if jc.suite_id:
        return suite_display_label(jc.suite_id, jc.suite_label)
    if jc.group == TELEMETRY_GROUP:
        return "Telemetry Infrastructure"
    return ""


def group_sort_key(group: str, category: Optional[str] = None) -> Tuple[int, str]:
    """Stable ordering: numbered suites by id, orchestration, preliminary, other."""
    cat_rank = {"numbered": 0, "preliminary": 1, "other": 2}.get(category or "", 3)
    m = re.match(r"^Suite (\d+)", group or "")
    if m:
        return (0, f"{int(m.group(1)):04d}")
    return (cat_rank * 10 + 1, group or "")


def ordered_groups(pairs: Iterable[Tuple[str, str]]) -> List[str]:
    """Order unique ``(group, category)`` pairs with :func:`group_sort_key`."""
    seen = {}
    for g, c in pairs:
        seen.setdefault(g, c)
    return sorted(seen, key=lambda g: group_sort_key(g, seen[g]))

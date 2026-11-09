"""Allocation (fairshare) accounting, waste attribution and fairshare-aware rightsizing.

Slurm's multifactor priority on MSI includes a **FairShare** factor that decays as the
account/user consumes *billed* usage.  Billing is driven by trackable resources (TRES):
with ``PriorityFlags=MAX_TRES`` the billed amount of a job is the **larger** of its
allocated cores and its memory expressed in "core equivalents"::

    mem_equiv_cores = req_mem_gb / mem_per_core_gb
    billed_cores    = max(req_cpus, mem_equiv_cores)
    billed_core_h   = billed_cores * elapsed_hours

``mem_per_core_gb`` is the memory that comes "for free" with one core.  On MSI Agate
``msismall`` the default ``--mem-per-cpu`` is 3900 MB, i.e. about **3.9 GB per core**
(:data:`DEFAULT_MEM_PER_CORE_GB`).  This is a documented *assumption*: the exact
``TRESBillingWeights`` of each partition can be checked on the cluster with
``scontrol show partition <name> | grep -i billing`` and overridden through
``mem_per_core_gb=`` / :data:`PARTITION_MEM_PER_CORE_GB`.

Billed usage only depends on *allocations* x *elapsed time* -- not on what the job actually
used -- so over-requested memory, idle cores and failed jobs all cost fairshare.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

#: Memory per billed core on msismall (default --mem-per-cpu=3900M ~= 3.9 GB).  Assumption.
DEFAULT_MEM_PER_CORE_GB = 3.9

#: Per-partition overrides of the memory-per-core billing ratio (GB per core).  Partitions
#: not listed use the ``mem_per_core_gb`` argument.  Values are assumptions to be checked
#: against ``scontrol show partition`` on the cluster.
PARTITION_MEM_PER_CORE_GB: Dict[str, float] = {
    "msismall": 3.9,
    "agsmall": 3.9,
    "msilarge": 3.9,
}

#: Default half-life of Slurm's usage decay (``PriorityDecayHalfLife``; Slurm default 7 days).
DEFAULT_DECAY_HALF_LIFE_DAYS = 7.0


def mem_per_core_series(df: pd.DataFrame, mem_per_core_gb: float = DEFAULT_MEM_PER_CORE_GB,
                        use_partition_overrides: bool = True) -> pd.Series:
    """GB-per-billed-core for each row (partition override or the global value)."""
    base = pd.Series(float(mem_per_core_gb), index=df.index)
    if use_partition_overrides and "partition" in df:
        over = df["partition"].map(PARTITION_MEM_PER_CORE_GB)
        # Only override when the caller kept the default (an explicit value applies everywhere)
        if math.isclose(mem_per_core_gb, DEFAULT_MEM_PER_CORE_GB):
            base = over.fillna(base)
    return base


def add_billing_columns(df: pd.DataFrame, mem_per_core_gb: float = DEFAULT_MEM_PER_CORE_GB) -> pd.DataFrame:
    """Return a copy of ``df`` with billed-usage and waste columns (all in core-hours).

    Columns added:

    * ``alloc_core_h``      -- ``req_cpus * elapsed_h``
    * ``mem_equiv_cores``   -- ``req_mem_gb / mem_per_core``
    * ``mem_equiv_core_h``  -- ``mem_equiv_cores * elapsed_h``
    * ``billed_cores``      -- ``max(req_cpus, mem_equiv_cores)``
    * ``billed_core_h``     -- ``billed_cores * elapsed_h`` (what fairshare is charged)
    * ``mem_dominated``     -- memory, not cores, set the bill
    * ``used_core_h``       -- ``alloc_core_h * cpu_eff`` (NaN when CPU efficiency unknown)
    * ``needed_cores``      -- ``max(effective cores, MaxRSS / mem_per_core)``; unknown CPU
      efficiency counts as fully used, unknown MaxRSS as fully used (conservative)
    * ``needed_core_h``     -- ``needed_cores * elapsed_h`` (minimum bill for the same work)
    * ``slack_core_h``      -- ``billed - needed`` for productive (COMPLETED etc.) jobs
    * ``failed_core_h``     -- full bill of unproductive jobs (OOM, FAILED, TIMEOUT, CANCELLED...)
    * ``useful_core_h``     -- ``needed_core_h`` of productive jobs
    * ``idle_cpu_core_h``   -- ``alloc_core_h * (1 - cpu_eff)`` where CPU efficiency is known
    * ``idle_mem_gb_h``     -- ``(req_mem - MaxRSS) * elapsed_h`` where MaxRSS is known
    """
    out = df.copy()
    mpc = mem_per_core_series(out, mem_per_core_gb)
    elapsed = out["elapsed_h"].fillna(0.0).clip(lower=0.0)
    cpus = out["req_cpus"].fillna(1).clip(lower=1).astype(float)
    mem = out["req_mem_gb"].fillna(0.0).clip(lower=0.0)
    out["mem_per_core_gb"] = mpc
    out["alloc_core_h"] = cpus * elapsed
    out["mem_equiv_cores"] = mem / mpc
    out["mem_equiv_core_h"] = out["mem_equiv_cores"] * elapsed
    out["billed_cores"] = np.maximum(cpus, out["mem_equiv_cores"])
    out["billed_core_h"] = out["billed_cores"] * elapsed
    out["mem_dominated"] = out["mem_equiv_cores"] > cpus

    cpu_frac = out["cpu_eff_clean"] / 100.0 if "cpu_eff_clean" in out else pd.Series(np.nan, index=out.index)
    out["used_core_h"] = out["alloc_core_h"] * cpu_frac
    eff_cores = (cpus * cpu_frac).fillna(cpus)
    rss = out["used_max_rss_gb"].where(out["used_max_rss_gb"] > 0)
    rss_cores = (rss / mpc).fillna(out["mem_equiv_cores"])
    out["needed_cores"] = np.minimum(np.maximum(eff_cores, rss_cores), out["billed_cores"])
    out["needed_core_h"] = out["needed_cores"] * elapsed

    unprod = out["unproductive"].astype(bool) if "unproductive" in out else pd.Series(False, index=out.index)
    out["failed_core_h"] = out["billed_core_h"].where(unprod, 0.0)
    out["useful_core_h"] = out["needed_core_h"].where(~unprod, 0.0)
    out["slack_core_h"] = (out["billed_core_h"] - out["needed_core_h"]).clip(lower=0).where(~unprod, 0.0)
    out["idle_cpu_core_h"] = out["alloc_core_h"] * (1.0 - cpu_frac)
    out["idle_mem_gb_h"] = ((mem - rss).clip(lower=0) * elapsed).where(rss.notna())
    return out


def usage_summary(billed: pd.DataFrame, by: Union[str, Sequence[str]] = "group") -> pd.DataFrame:
    """Aggregate billed usage and waste per group.

    Returns one row per group with job counts, core-hour totals and fractions:
    ``slack_frac`` (allocated-but-unneeded share of the bill), ``failed_frac`` (share spent
    on unproductive jobs), ``wasted_frac`` (= slack + failed), ``mem_dominated_frac`` (share
    of the bill from jobs whose memory request, not core count, set the price), and the
    share of the total bill (``share_of_total``).
    """
    by = [by] if isinstance(by, str) else list(by)
    cols = ["alloc_core_h", "mem_equiv_core_h", "billed_core_h", "useful_core_h",
            "slack_core_h", "failed_core_h", "idle_cpu_core_h", "idle_mem_gb_h"]
    if billed.empty:
        return pd.DataFrame(columns=by + ["jobs", "unproductive_jobs"] + cols)
    g = billed.groupby(by, dropna=False, sort=False)
    out = g[cols].sum(min_count=1)
    out.insert(0, "jobs", g.size())
    out.insert(1, "unproductive_jobs", g["unproductive"].sum().astype(int))
    out["mem_dominated_core_h"] = billed["billed_core_h"].where(billed["mem_dominated"], 0.0).groupby(
        [billed[c] for c in by], dropna=False, sort=False).sum()
    denom = out["billed_core_h"].replace(0, np.nan)
    out["slack_frac"] = out["slack_core_h"] / denom
    out["failed_frac"] = out["failed_core_h"] / denom
    out["wasted_frac"] = (out["slack_core_h"] + out["failed_core_h"]) / denom
    out["mem_dominated_frac"] = out["mem_dominated_core_h"] / denom
    total = billed["billed_core_h"].sum()
    out["share_of_total"] = out["billed_core_h"] / total if total > 0 else np.nan
    return out


def failure_usage(billed: pd.DataFrame, by: Union[str, Sequence[str]] = "group") -> pd.DataFrame:
    """Billed core-hours of unproductive jobs broken down by terminal state."""
    sub = billed[billed["unproductive"]]
    by = [by] if isinstance(by, str) else list(by)
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index=by, columns="state_base", values="billed_core_h",
                           aggfunc="sum", fill_value=0.0)


def round_up_mem(gb: float, step_small: float = 4.0, step_large: float = 16.0, threshold: float = 64.0) -> float:
    """Round a memory request up to a clean value (4 GB steps up to 64 GB, then 16 GB)."""
    if not np.isfinite(gb) or gb <= 0:
        return step_small
    step = step_small if gb <= threshold else step_large
    return float(math.ceil(gb / step) * step)


def rightsizing_table(
    billed: pd.DataFrame,
    by: Sequence[str] = ("group", "job_name_resolved", "req_mem_gb"),
    min_completed: int = 5,
    mem_headroom: float = 1.2,
    cpu_headroom: float = 1.25,
    mem_per_core_gb: float = DEFAULT_MEM_PER_CORE_GB,
    oom_keys: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Fairshare-aware request recommendations per job type / request profile.

    The default key is a *request profile* (group, job name, requested memory): one job
    name is often reused for different workloads (e.g. a 3-minute and a 3-hour version of a
    sweep), each submitted with its own ``--mem``.  OOM evidence is pooled over the whole job
    name (``oom_keys``), so a profile is never recommended below a request that OOM'd.

    For each ``by`` key with at least ``min_completed`` COMPLETED jobs that have MaxRSS:

    * **memory**: ``max(headroom x max completed MaxRSS, next step above the largest request
      that ended OUT_OF_MEMORY)``, rounded up with :func:`round_up_mem`.  Memory up to
      ``cpus x mem_per_core`` is free (it does not raise the bill), so the recommendation is
      never pushed below what would make cores the billing driver anyway.
    * **cores**: ``ceil(headroom x P95 effective cores)`` (effective cores =
      ``req_cpus x CPU efficiency``) when CPU efficiency is known for at least
      ``min_completed`` completed jobs, else unchanged.  Never more than requested.
    * **counterfactual bill**: completed jobs re-priced at the recommended request with the
      same elapsed time (a fewer-core job may run longer if it actually used its threads;
      the P95-of-effective-cores rule keeps that risk small).

    Recommendations raise memory where jobs OOM'd (``action`` = "increase") -- failures
    cost the whole bill, so that is usually the cheaper option overall.
    """
    by = list(by)
    rows: List[dict] = []
    if billed.empty:
        return pd.DataFrame()
    if oom_keys is None:
        oom_keys = [c for c in by if c not in ("req_mem_gb", "req_cpus")] or list(by)
    oom_keys = list(oom_keys)
    oom_rows = billed[(billed["state_base"] == "OUT_OF_MEMORY") & (billed["req_mem_gb"] > 0)]
    oom_floor_map = oom_rows.groupby(oom_keys)["req_mem_gb"].max().to_dict() if not oom_rows.empty else {}
    oom_count_map = oom_rows.groupby(oom_keys).size().to_dict() if not oom_rows.empty else {}
    for key, sub in billed.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        kd = dict(zip(by, key))
        okey = tuple(kd[c] for c in oom_keys)
        okey = okey[0] if len(okey) == 1 else okey
        comp = sub[sub["state_base"] == "COMPLETED"]
        comp_rss = comp[comp["used_max_rss_gb"] > 0]
        if len(comp_rss) < min_completed:
            continue
        mpc = float(np.nanmedian(sub["mem_per_core_gb"])) if "mem_per_core_gb" in sub else mem_per_core_gb
        cur_mem = float(comp["req_mem_gb"].median())
        cur_cpus = int(round(comp["req_cpus"].median()))
        rss_max = float(comp_rss["used_max_rss_gb"].max())
        rss_p95 = float(comp_rss["used_max_rss_gb"].quantile(0.95))
        oom_floor = float(oom_floor_map.get(okey, 0.0))
        target = max(rss_max * mem_headroom, (oom_floor + 1e-6) if oom_floor > 0 else 0.0)
        rec_mem = round_up_mem(target)

        cpu_known = comp[comp["cpu_eff_clean"].notna()]
        if len(cpu_known) >= min_completed:
            eff_cores = cpu_known["req_cpus"].clip(lower=1) * cpu_known["cpu_eff_clean"] / 100.0
            p95_cores = float(eff_cores.quantile(0.95))
            rec_cpus = int(min(max(1, math.ceil(p95_cores * cpu_headroom)), max(cur_cpus, 1)))
            med_cpu_eff = float(cpu_known["cpu_eff_clean"].median())
        else:
            p95_cores, rec_cpus, med_cpu_eff = np.nan, max(cur_cpus, 1), np.nan
        # memory below cpus*mpc is free: don't recommend less than that (no saving, more OOM risk)
        free_mem = float(math.floor(min(cur_mem, rec_cpus * mpc)))
        if rec_mem < free_mem:
            rec_mem = free_mem

        elapsed = comp["elapsed_h"].clip(lower=0)
        billed_now = float(comp["billed_core_h"].sum())
        billed_rec = float((np.maximum(rec_cpus, rec_mem / mpc) * elapsed).sum())
        if rec_mem > cur_mem + 1e-9:
            action = "increase (OOM risk)" if oom_floor >= cur_mem else "increase"
        elif billed_rec < billed_now * 0.99:
            action = "decrease"
        else:
            action = "keep"
        rows.append({
            **kd,
            "jobs": len(sub),
            "completed": len(comp),
            "oom": int((sub["state_base"] == "OUT_OF_MEMORY").sum()),
            "oom_job_name": int(oom_count_map.get(okey, 0)),
            "req_mem_gb": cur_mem,
            "req_cpus": cur_cpus,
            "rss_p95_gb": rss_p95,
            "rss_max_gb": rss_max,
            "oom_floor_gb": oom_floor,
            "median_cpu_eff_pct": med_cpu_eff,
            "p95_eff_cores": p95_cores,
            "rec_mem_gb": rec_mem,
            "rec_cpus": rec_cpus,
            "billed_now_core_h": billed_now,
            "billed_rec_core_h": billed_rec,
            "saving_core_h": billed_now - billed_rec,
            "saving_frac": (billed_now - billed_rec) / billed_now if billed_now > 0 else np.nan,
            "failed_core_h": float(sub["failed_core_h"].sum()),
            "action": action,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("saving_core_h", ascending=False).reset_index(drop=True)
    return out


def usage_rate_grid(billed: pd.DataFrame, freq_hours: float = 1.0):
    """Billed core-hours consumed in each time bin (jobs spread over their run interval).

    Returns ``(edges, usage)`` where ``usage[i]`` is the billed core-hours consumed during
    ``[edges[i], edges[i+1])``.
    """
    run = billed.dropna(subset=["start_dt", "end_dt"])
    run = run[(run["end_dt"] > run["start_dt"]) & (run["billed_cores"] > 0)]
    if run.empty:
        return np.array([], dtype="datetime64[ns]"), np.array([])
    step = pd.Timedelta(hours=freq_hours)
    t0 = run["start_dt"].min().floor("h")
    t1 = run["end_dt"].max().ceil("h") + step
    n = int((t1 - t0) / step) + 1
    edges = t0 + step * np.arange(n + 1)
    s = ((run["start_dt"] - t0) / step).to_numpy(float)
    e = ((run["end_dt"] - t0) / step).to_numpy(float)
    rate = run["billed_cores"].to_numpy(float) * freq_hours  # core-hours per full bin
    # integral of a box function via cumulative coverage: F(x) = sum rate*clip(x-s,0,e-s)
    diff = np.zeros(n + 2)
    i_s, i_e = np.floor(s).astype(int), np.floor(e).astype(int)
    f_s, f_e = s - i_s, e - i_e
    # full-bin rate between i_s and i_e, with fractional corrections at both ends
    np.add.at(diff, i_s, rate)
    np.add.at(diff, i_e, -rate)
    usage = np.cumsum(diff)[:n].astype(float)
    corr = np.zeros(n + 1)
    np.add.at(corr, i_s, -rate * f_s)
    np.add.at(corr, i_e, rate * f_e)
    usage = usage + corr[:n]
    return edges, np.clip(usage, 0, None)


def prior_usage_at(
    billed: pd.DataFrame,
    times: Iterable,
    window_days: float = 7.0,
    half_life_days: Optional[float] = DEFAULT_DECAY_HALF_LIFE_DAYS,
    freq_hours: float = 1.0,
) -> pd.DataFrame:
    """Usage consumed *before* each time in ``times``.

    Returns a DataFrame (index aligned with ``times``) with
    ``window_core_h`` -- billed core-hours consumed in ``[t - window_days, t)`` -- and
    ``decayed_core_h`` -- all prior usage weighted by ``0.5 ** (age / half_life)``, which is
    how Slurm ages usage for FairShare (``PriorityDecayHalfLife``).
    """
    times = pd.to_datetime(pd.Series(list(times)))
    edges, usage = usage_rate_grid(billed, freq_hours)
    out = pd.DataFrame(index=times.index, data={"window_core_h": np.nan, "decayed_core_h": np.nan})
    if len(usage) == 0 or times.empty:
        return out
    x_edges = (edges - edges[0]) / pd.Timedelta(hours=freq_hours)
    cum = np.concatenate([[0.0], np.cumsum(usage)])  # cumulative usage at each edge
    x = ((times - edges[0]) / pd.Timedelta(hours=freq_hours)).to_numpy(float)
    w = window_days * 24.0 / freq_hours
    c_now = np.interp(x, x_edges, cum, left=0.0, right=cum[-1])
    c_then = np.interp(x - w, x_edges, cum, left=0.0, right=cum[-1])
    out["window_core_h"] = c_now - c_then
    if half_life_days:
        lam = math.log(2) / (half_life_days * 24.0 / freq_hours)
        decay = math.exp(-lam)
        dec = np.zeros(len(usage) + 1)
        for i, u in enumerate(usage):  # value at edge i+1 (end of bin i)
            dec[i + 1] = dec[i] * decay + u
        idx = np.clip(np.floor(x).astype(int), 0, len(dec) - 1)
        frac = np.clip(x - idx, 0, None)
        val = dec[idx] * np.exp(-lam * frac)
        val[x <= 0] = 0.0
        beyond = x > len(usage)
        if beyond.any():
            val[beyond] = dec[-1] * np.exp(-lam * (x[beyond] - len(usage)))
        out["decayed_core_h"] = val
    return out


def submission_events(billed: pd.DataFrame) -> pd.DataFrame:
    """Collapse array tasks into submission events (same job name + submit time).

    Columns: ``group``, ``category``, ``job_name_resolved``, ``submit_dt``, ``tasks``,
    ``first_wait_min`` (time until the first task started -- the scheduler's response to
    the submission, unaffected by array throttling), ``median_wait_min``,
    ``req_mem_gb``, ``req_cpus``, ``billed_cores`` (per task, median).
    """
    sub = billed.dropna(subset=["submit_dt"])
    sub = sub[sub["queue_wait_min_clean"].notna()]
    if sub.empty:
        return pd.DataFrame(columns=["group", "category", "job_name_resolved", "submit_dt", "tasks",
                                     "first_wait_min", "median_wait_min", "req_mem_gb", "req_cpus",
                                     "billed_cores"])
    g = sub.groupby(["job_name_resolved", "submit_dt"], sort=False)
    ev = g.agg(
        group=("group", "first"), category=("category", "first"), tasks=("job_id", "size"),
        first_wait_min=("queue_wait_min_clean", "min"), median_wait_min=("queue_wait_min_clean", "median"),
        req_mem_gb=("req_mem_gb", "median"), req_cpus=("req_cpus", "median"),
        billed_cores=("billed_cores", "median"),
    ).reset_index()
    return ev.sort_values("submit_dt").reset_index(drop=True)


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    """Spearman rank correlation without SciPy (NaNs dropped)."""
    a = pd.Series(list(x), dtype=float)
    b = pd.Series(list(y), dtype=float)
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return float("nan")
    ra, rb = a[m].rank(), b[m].rank()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


__all__ = [
    "DEFAULT_MEM_PER_CORE_GB", "PARTITION_MEM_PER_CORE_GB", "DEFAULT_DECAY_HALF_LIFE_DAYS",
    "add_billing_columns", "usage_summary", "failure_usage", "round_up_mem",
    "rightsizing_table", "usage_rate_grid", "prior_usage_at", "submission_events", "spearman",
]

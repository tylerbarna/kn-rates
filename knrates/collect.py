"""Compile all analysis results into one version-controlled snapshot file.

The snapshot is a single ZIP archive (``snapshots/knrates_results_latest.zip`` by
default) containing one CSV per table plus ``manifest.json`` (provenance, source
fingerprints, validation report, row counts and SHA-256 checksums).  Notebooks
load it with :func:`knrates.load_snapshot`, so they run anywhere -- also off the
cluster where ``results/`` is not available.

Usage (from the repository root)::

    python -m knrates.collect                  # rebuild and replace the 'latest' snapshot
    python -m knrates.collect --timestamped    # write snapshots/knrates_results_<UTC>.zip (never overwrites)
    python -m knrates.collect --timestamped --also-latest
    python -m knrates.collect --dry-run        # build + validate only, write nothing
    python -m knrates.collect --force          # ignore the per-suite cache and re-read everything

Safety
------
* Every table is validated (schema, finiteness, n_det <= n_sim, VT >= 0,
  consistency between stored and recomputed efficiencies, duplicate runs,
  suite run counts not shrinking relative to the previous snapshot, ...).
  Any *error* aborts the build and leaves existing snapshots untouched
  (exit status 1); *warnings* are recorded in the manifest.
* The archive is written to a temporary file, re-opened and verified
  (checksums + row counts) and only then atomically moved into place.
* Unparseable / truncated result files (e.g. from jobs that died mid-write) are
  skipped and listed in the manifest instead of poisoning the tables.
* Suites whose source files are unchanged since the previous snapshot are
  re-used from it (fingerprint = file count, total size, newest mtime), so a
  routine update only re-reads what changed.

Adding a new suite
------------------
Write a ``collect_<name>(ctx) -> dict[str, DataFrame]`` function, decorate it with
``@register(<fingerprint globs>)`` and add any table-specific checks to
``validate_tables``.  Generic simulation-result directories can reuse
:func:`collect_simulation_suite`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results"
SNAPSHOT_DIR = REPO_ROOT / "snapshots"
LATEST_NAME = "knrates_results_latest.zip"
STAMP_FMT = "%Y%m%dT%H%M%SZ"

# Detection-criteria keys used throughout the project
CRITERIA_KEYS = [
    "snr_threshold", "snr_threshold_secondary", "min_detections", "min_detections_primary",
    "max_timespan_days", "min_time_separation_hours", "min_rise_rate", "min_fade_rate",
    "min_galactic_lat", "require_fast_transient",
]

# Suite 02 fiducial grid point (must match analysis_suites/02_granular_criteria_sweep/run_granular_sweep.py)
SUITE02_FIDUCIAL = {
    "snr_threshold": 5.0, "snr_threshold_secondary": 3.0, "min_detections": 2,
    "min_detections_primary": 1, "max_timespan_days": 14.0, "min_time_separation_hours": 3.0,
    "min_rise_rate": 0.0, "min_fade_rate": 0.30, "min_galactic_lat": 15.0,
    "require_fast_transient": True,
}


# ---------------------------------------------------------------------------
# Context / registry
# ---------------------------------------------------------------------------
@dataclass
class Context:
    results_dir: Path
    previous: Optional["object"] = None  # knrates.snapshot.Snapshot of the previous build
    force: bool = False
    max_source_runs: int = 10
    warnings: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    fingerprints: Dict[str, dict] = field(default_factory=dict)
    reused: List[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [warn] {msg}", flush=True)


@dataclass
class Collector:
    name: str
    func: Callable[[Context], Dict[str, pd.DataFrame]]
    globs: Sequence[str]


REGISTRY: List[Collector] = []


def register(*globs: str):
    def deco(func):
        REGISTRY.append(Collector(func.__name__.replace("collect_", ""), func, globs))
        return func
    return deco


def _fingerprint(results_dir: Path, patterns: Sequence[str]) -> dict:
    files = []
    for pat in patterns:
        files.extend(glob.glob(str(results_dir / pat), recursive=True))
    files = sorted(set(files))
    size = 0
    mtime = 0.0
    for f in files:
        st = os.stat(f)
        size += st.st_size
        mtime = max(mtime, st.st_mtime)
    return {"n_files": len(files), "total_bytes": size, "newest_mtime": round(mtime, 3)}


def _rel(path: Path, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Generic simulation-result parsing (survey_sim SimulationResult JSON)
# ---------------------------------------------------------------------------
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})")
_TASK_RE = re.compile(r"_task_([0-9]+)")


def _load_json(path: Path, ctx: Context) -> Optional[dict]:
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception as exc:  # truncated/corrupt output from a failed job
        ctx.skipped_files.append(f"{_rel(path, ctx.results_dir)}: {type(exc).__name__}: {exc}"[:300])
        return None


def _source_record(src: dict) -> dict:
    t = src.get("photometry_times") or []
    m = src.get("photometry_mags") or []
    b = src.get("photometry_bands") or []
    tp = src.get("true_params") or {}
    incl = tp.get("inclination_EM", tp.get("inclination_em", np.nan))
    t = [float(x) for x in t]
    m = [float(x) for x in m]
    t_exp = float(src.get("t_exp", np.nan))
    rec = {
        "z": float(src.get("z", np.nan)),
        "t_exp_mjd": t_exp,
        "inclination_rad": float(incl) if incl is not None else np.nan,
        "n_phot": len(t),
        "n_bands": len(set(b)),
        "phase_first_det_d": (min(t) - t_exp) if t else np.nan,
        "span_det_d": (max(t) - min(t)) if t else np.nan,
        "mag_brightest": min(m) if m else np.nan,
    }
    return rec


def collect_simulation_suite(ctx: Context, suite_id: str, subdir: str,
                             per_source_models: Sequence[str] = (),
                             max_source_runs: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Parse ``results/<subdir>/<model>/*.json`` (+ ``.metadata.json``) files.

    Returns tables ``runs`` (one row per realisation), ``sources`` (selected
    per-detected-source quantities; all runs for ``per_source_models``, otherwise
    the first ``max_source_runs`` runs by task id), ``texp_month`` (detections per
    explosion month, all runs) and ``z_hist`` (detections per redshift bin, all runs).
    """
    base = ctx.results_dir / subdir
    max_source_runs = ctx.max_source_runs if max_source_runs is None else max_source_runs
    files = sorted(f for f in base.glob("*/*.json") if not f.name.endswith(".metadata.json"))
    runs, sources, texp_rows, z_rows = [], [], [], []
    z_edges = np.round(np.arange(0.0, 0.3001, 0.0025), 4)
    month_edges = None
    per_model_count: Dict[str, int] = {}
    for f in files:
        meta_p = Path(f"{f}.metadata.json")
        meta = {}
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text())
            except Exception as exc:
                ctx.warn(f"unreadable metadata {_rel(meta_p, ctx.results_dir)}: {exc}")
        data = _load_json(f, ctx)
        if data is None:
            continue
        rs_list = data.get("rate_summaries") or []
        if len(rs_list) != 1:
            ctx.skipped_files.append(f"{_rel(f, ctx.results_dir)}: expected 1 rate summary, found {len(rs_list)}")
            continue
        rs = rs_list[0]
        model = meta.get("model") or f.parent.name
        task = meta.get("task_id")
        if task is None:
            mt = _TASK_RE.search(f.name)
            task = mt.group(1) if mt else None
        ts = _TS_RE.search(f.name)
        n_sim = int(rs.get("n_simulated", data.get("n_simulated", 0)))
        n_det = int(rs.get("n_detected", data.get("n_detected", 0)))
        dets = data.get("detected_sources") or []
        params = meta.get("parameters") or meta.get("params") or {}
        if not params and dets:
            # parameters are not in the metadata: take them from the recovered sources and
            # mark any that differ between sources (e.g. isotropic inclinations) as "varied"
            vals: Dict[str, set] = {}
            for s_ in dets:
                for k_, v_ in (s_.get("true_params") or {}).items():
                    vals.setdefault(k_, set()).add(v_)
            params = {k_: (next(iter(v_)) if len(v_) == 1 else "varied") for k_, v_ in vals.items()}
        crit = meta.get("detection_criteria") or {}
        row = {
            "suite": suite_id,
            "model": model,
            "model_family": meta.get("model_family", "Metzger" if "Metzger" in model else "Bu2026"),
            "task_id": None if task is None else str(task),
            "run_index": meta.get("run_index"),
            "seed": meta.get("seed"),
            "file": _rel(f, ctx.results_dir),
            "file_timestamp": ts.group(1) if ts else None,
            "n_simulated": n_sim,
            "n_detected": n_det,
            "n_detected_sources_listed": len(dets),
            "efficiency": n_det / n_sim if n_sim else np.nan,
            "efficiency_file": float(rs.get("overall_efficiency", np.nan)),
            "vt_gpc3_yr": float(rs.get("effective_vt_gpc3_yr", np.nan)),
            "z_max": float(rs.get("z_max", meta.get("z_max", np.nan))),
            "survey_omega_sr": float(rs.get("survey_omega_sr", np.nan)),
            "survey_duration_years": float(rs.get("survey_duration_years", np.nan)),
            "injected_rate": float(rs.get("volumetric_rate", meta.get("rate", np.nan))),
            "randomize_params": bool(meta.get("randomize_params", False)),
            "elapsed_seconds": meta.get("elapsed_seconds"),
            "params_json": json.dumps(params, sort_keys=True),
            "criteria_json": json.dumps(crit, sort_keys=True),
        }
        for k in CRITERIA_KEYS:
            row[f"crit_{k}"] = crit.get(k, np.nan)
        runs.append(row)

        # aggregate histograms over all runs
        recs = [_source_record(s) for s in dets]
        if recs:
            zz = np.array([r["z"] for r in recs])
            tt = np.array([r["t_exp_mjd"] for r in recs])
            zc, _ = np.histogram(zz, bins=z_edges)
            z_rows.append(pd.DataFrame({"suite": suite_id, "model": model, "task_id": row["task_id"],
                                        "z_lo": z_edges[:-1], "z_hi": z_edges[1:], "n_detected": zc}))
            # calendar months by explosion epoch
            months = pd.to_datetime(tt - 40587.0, unit="D").to_period("M").astype(str)
            vc = pd.Series(months).value_counts()
            texp_rows.append(pd.DataFrame({"suite": suite_id, "model": model, "task_id": row["task_id"],
                                           "month": vc.index, "n_detected": vc.values}))
        keep_sources = model in per_source_models or per_model_count.get(model, 0) < max_source_runs
        if keep_sources and recs:
            per_model_count[model] = per_model_count.get(model, 0) + 1
            sdf = pd.DataFrame(recs)
            sdf.insert(0, "task_id", row["task_id"])
            sdf.insert(0, "model", model)
            sdf.insert(0, "suite", suite_id)
            sources.append(sdf)
    out = {
        "runs": pd.DataFrame(runs),
        "sources": pd.concat(sources, ignore_index=True) if sources else pd.DataFrame(),
        "z_hist": (pd.concat(z_rows).groupby(["suite", "model", "z_lo", "z_hi"], as_index=False)["n_detected"].sum()
                   if z_rows else pd.DataFrame()),
        "texp_month": (pd.concat(texp_rows).groupby(["suite", "model", "month"], as_index=False)["n_detected"].sum()
                       if texp_rows else pd.DataFrame()),
    }
    return out


# Populations that were intentionally retired; their disappearance is not a partial write.
# Suite 03 "BroadPrior_*" drew ONE parameter set per realisation and were replaced by the per-transient
# BroadPopulation_Isotropic (compare_priors_at2017gfo.py, Nov 2026).
RETIRED_POPULATIONS = {"suite03_runs": {"BroadPrior_FixedView", "BroadPrior_Isotropic"}}

# Bump a suite's version whenever its collector output changes, so cached tables are rebuilt.
COLLECTOR_VERSIONS = {"suite01": 1, "suite07": 1, "suite03": 2, "suite02": 2, "models": 1}


def _reuse_or_run(ctx: Context, name: str, patterns: Sequence[str],
                  build: Callable[[], Dict[str, pd.DataFrame]], table_prefix: str) -> Dict[str, pd.DataFrame]:
    fp = _fingerprint(ctx.results_dir, patterns)
    fp["collector_version"] = COLLECTOR_VERSIONS.get(name, 1)
    ctx.fingerprints[name] = fp
    prev = ctx.previous
    if (not ctx.force and prev is not None and prev.manifest.get("fingerprints", {}).get(name) == fp
            and fp["n_files"] > 0):
        tables = {t: prev.tables[t] for t in prev.tables if t.startswith(table_prefix)}
        if tables:
            ctx.reused.append(name)
            print(f"  [cache] {name}: sources unchanged ({fp['n_files']} files) -> re-using previous tables", flush=True)
            return {t[len(table_prefix):]: df for t, df in tables.items()}
    if fp["n_files"] == 0:
        return {}
    return build()


# ---------------------------------------------------------------------------
# Suite collectors
# ---------------------------------------------------------------------------
@register("01_large_scale_rate_limits/*/*.json")
def collect_suite01(ctx: Context) -> Dict[str, pd.DataFrame]:
    """Suite 01: fiducial-parameter ensembles (Bu2026Fixed, Bu2026Vary, Metzger)."""
    return _reuse_or_run(ctx, "suite01", ["01_large_scale_rate_limits/*/*.json"],
                         lambda: collect_simulation_suite(ctx, "01", "01_large_scale_rate_limits",
                                                          per_source_models=("Bu2026Vary",)),
                         "suite01_")


@register("07_gw170817_rate_limits/*/*.json")
def collect_suite07(ctx: Context) -> Dict[str, pd.DataFrame]:
    """Suite 07: GW170817/AT2017gfo-pinned ensembles (Bu2026 fixed/isotropic, Metzger blue/mean)."""
    return _reuse_or_run(ctx, "suite07", ["07_gw170817_rate_limits/*/*.json"],
                         lambda: collect_simulation_suite(ctx, "07", "07_gw170817_rate_limits",
                                                          per_source_models=("GW170817_Bu2026_Isotropic",)),
                         "suite07_")


@register("03_prior_vs_at2017gfo/*")
def collect_suite03(ctx: Context) -> Dict[str, pd.DataFrame]:
    """Suite 03: criteria regime x population (AT2017gfo-like vs. single random prior draws)."""

    def build():
        base = ctx.results_dir / "03_prior_vs_at2017gfo"
        summ = base / "prior_vs_at2017gfo_summary.csv"
        if not summ.exists():
            return {}
        df = pd.read_csv(summ)
        # Record the actual parameters used in each run (taken from the detected
        # sources' true_params; BroadPrior runs use ONE draw per run).
        pj, zmax = [], []
        for _, r in df.iterrows():
            f = base / f"sim_{r['criteria_regime']}_{r['population_type']}_run{int(r['run_idx'])}.json"
            params, zm = {}, np.nan
            if f.exists():
                d = _load_json(f, ctx)
                if d:
                    dets = d.get("detected_sources") or []
                    if dets:
                        params = dets[0].get("true_params", {})
                    rs = (d.get("rate_summaries") or [{}])[0]
                    zm = rs.get("z_max", np.nan)
            pj.append(json.dumps(params, sort_keys=True))
            zmax.append(zm)
        df["params_json"] = pj
        df["z_max"] = zmax
        df = df.rename(columns={"effective_vt_gpc3_yr": "vt_gpc3_yr"})
        df["suite"] = "03"
        return {"runs": df}

    return _reuse_or_run(ctx, "suite03", ["03_prior_vs_at2017gfo/*"], build, "suite03_")


# Named criteria regimes evaluated from the full Suite 02 grid (|b| is fixed at the fiducial value in the grid).
SUITE02_NAMED_CONFIGS = {
    "Ultra-relaxed": dict(snr_threshold=4.0, snr_threshold_secondary=2.5, max_timespan_days=28.0, min_fade_rate=0.0),
    "Relaxed (S03-like)": dict(snr_threshold=4.0, snr_threshold_secondary=2.5, max_timespan_days=28.0,
                               min_time_separation_hours=1.0, min_fade_rate=0.1),
    "Fiducial": dict(),
    "N_min = 3": dict(min_detections=3),
    "Strict (S03-like)": dict(snr_threshold=6.0, snr_threshold_secondary=4.0, min_detections=3,
                              min_detections_primary=2, max_timespan_days=7.0, min_time_separation_hours=6.0,
                              min_fade_rate=0.5),
    "Ultra-strict": dict(snr_threshold=6.0, snr_threshold_secondary=4.0, min_detections=3, max_timespan_days=7.0,
                         min_fade_rate=0.6),
}

_S02_USECOLS = CRITERIA_KEYS + ["model", "n_detected", "n_simulated", "overall_efficiency",
                                "overall_efficiency_std", "effective_vt_gpc3_yr",
                                "effective_vt_gpc3_yr_std", "rate_upper_95", "n_samples"]


def _suite02_tables(df: pd.DataFrame, model: str) -> Dict[str, pd.DataFrame]:
    fid = SUITE02_FIDUCIAL
    df = df.copy()
    df["require_fast_transient"] = df["require_fast_transient"].astype(str).str.lower().isin(["true", "1"])

    def at_fid(frame, free):
        m = np.ones(len(frame), dtype=bool)
        for k, v in fid.items():
            if k in free:
                continue
            col = frame[k]
            m &= (col == v) if isinstance(v, bool) else np.isclose(col.astype(float), float(v))
        return frame[m]

    keep = CRITERIA_KEYS + ["overall_efficiency", "overall_efficiency_std", "effective_vt_gpc3_yr",
                            "effective_vt_gpc3_yr_std", "rate_upper_95", "n_samples"]
    fid_row = at_fid(df, ())[keep].assign(model=model)
    slices1d = []
    for k in CRITERIA_KEYS:
        s = at_fid(df, (k,))[keep].assign(model=model, varied=k)
        slices1d.append(s)
    pairs = []
    for i, a in enumerate(CRITERIA_KEYS):
        for b in CRITERIA_KEYS[i + 1:]:
            s = at_fid(df, (a, b))[keep].assign(model=model, varied_a=a, varied_b=b)
            pairs.append(s)
    marg = []
    for k in CRITERIA_KEYS:
        g = df.groupby(k)["overall_efficiency"].agg(["mean", "median", "min", "max", "size"]).reset_index()
        g = g.rename(columns={k: "value"}).assign(model=model, criterion=k)
        g["value"] = g["value"].astype(float)
        marg.append(g)
    named = []
    for label, over in SUITE02_NAMED_CONFIGS.items():
        cfg = dict(fid, **over)
        m = np.ones(len(df), dtype=bool)
        for k, v in cfg.items():
            m &= (df[k] == v) if isinstance(v, bool) else np.isclose(df[k].astype(float), float(v))
        row = df[m][keep]
        if len(row) == 1:
            named.append(row.assign(model=model, regime=label))
    return {
        "named_configs": pd.concat(named, ignore_index=True) if named else pd.DataFrame(),
        "fiducial": fid_row,
        "slices_1d": pd.concat(slices1d, ignore_index=True),
        "slices_2d": pd.concat(pairs, ignore_index=True),
        "grid_marginals": pd.concat(marg, ignore_index=True),
        "grid_size": pd.DataFrame([{"model": model, "n_grid_points": len(df),
                                    "n_samples_min": int(df["n_samples"].min()),
                                    "n_samples_max": int(df["n_samples"].max())}]),
    }


@register("02_granular_criteria_sweep/granular_sweep_metrics_*.csv")
def collect_suite02(ctx: Context) -> Dict[str, pd.DataFrame]:
    """Suite 02: criteria sweep -> fiducial point, 1D/2D slices through the fiducial, grid marginals."""

    def build():
        base = ctx.results_dir / "02_granular_criteria_sweep"
        parts: Dict[str, List[pd.DataFrame]] = {}
        for f in sorted(base.glob("granular_sweep_metrics_*.csv")):
            model = f.stem.replace("granular_sweep_metrics_", "")
            print(f"    reading {f.name} ...", flush=True)
            dtypes = {k: "float32" for k in CRITERIA_KEYS if k != "require_fast_transient"}
            try:
                df = pd.read_csv(f, usecols=lambda c: c in _S02_USECOLS, dtype=dtypes)
            except Exception as exc:
                ctx.skipped_files.append(f"{_rel(f, ctx.results_dir)}: {exc}"[:300])
                continue
            for k in ("min_detections", "min_detections_primary"):
                df[k] = df[k].round().astype(int)
            for name, t in _suite02_tables(df, model).items():
                parts.setdefault(name, []).append(t)
        meta_rows = []
        for mp in sorted(base.glob("candidates/*/*.metadata.json")):
            try:
                m = json.loads(mp.read_text())
            except Exception:
                continue
            meta_rows.append({"model": m.get("model"), "file": _rel(mp, ctx.results_dir),
                              "seed": m.get("seed"), "n_transients": m.get("n_transients"),
                              "z_max": (m.get("params") or {}).get("z_max"),
                              "loose_vt_gpc3_yr": m.get("effective_vt_gpc3_yr"),
                              "loose_efficiency": m.get("overall_efficiency"),
                              "loose_criteria_json": json.dumps(m.get("loose_criteria", {}), sort_keys=True),
                              "params_json": json.dumps(m.get("params", {}), sort_keys=True)})
        out = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
        if meta_rows:
            out["candidate_runs"] = pd.DataFrame(meta_rows)
        return out

    return _reuse_or_run(ctx, "suite02",
                         ["02_granular_criteria_sweep/granular_sweep_metrics_*.csv",
                          "02_granular_criteria_sweep/candidates/*/*.metadata.json"],
                         build, "suite02_")


@register("model_lightcurves/*.csv")
def collect_models(ctx: Context) -> Dict[str, pd.DataFrame]:
    """Model light curves tabulated by scripts/make_model_lightcurves.py (absolute magnitudes vs. rest-frame time)."""

    def build():
        files = sorted((ctx.results_dir / "model_lightcurves").glob("*.csv"))
        frames = []
        for f in files:
            try:
                frames.append(pd.read_csv(f).assign(source_file=f.name))
            except Exception as exc:
                ctx.skipped_files.append(f"{_rel(f, ctx.results_dir)}: {exc}"[:300])
        return {"lightcurves": pd.concat(frames, ignore_index=True)} if frames else {}

    return _reuse_or_run(ctx, "models", ["model_lightcurves/*.csv"], build, "models_")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {
    "suite01_runs": ["model", "task_id", "n_simulated", "n_detected", "efficiency", "vt_gpc3_yr",
                     "survey_duration_years", "z_max"],
    "suite07_runs": ["model", "task_id", "n_simulated", "n_detected", "efficiency", "vt_gpc3_yr",
                     "survey_duration_years", "z_max"],
    "suite03_runs": ["criteria_regime", "population_type", "n_simulated", "n_detected", "vt_gpc3_yr"],
    "suite02_fiducial": ["model", "overall_efficiency", "effective_vt_gpc3_yr"],
}


def _check_runs(name: str, df: pd.DataFrame, errors: List[str], warnings: List[str]) -> pd.DataFrame:
    if df.empty:
        errors.append(f"{name}: table is empty")
        return df
    bad = df["n_detected"] > df["n_simulated"]
    if bad.any():
        errors.append(f"{name}: {int(bad.sum())} rows with n_detected > n_simulated")
    if (df["n_simulated"] <= 0).any():
        errors.append(f"{name}: non-positive n_simulated")
    vt = df["vt_gpc3_yr"].astype(float)
    if (~np.isfinite(vt)).any() or (vt < 0).any():
        errors.append(f"{name}: non-finite or negative VT values")
    if "efficiency_file" in df:
        diff = (df["efficiency"] - df["efficiency_file"]).abs()
        # files store efficiency rounded to 5 decimals
        if (diff > 1e-5 + 1e-9).any():
            errors.append(f"{name}: stored and recomputed efficiencies disagree in {int((diff > 1e-5).sum())} rows")
    if "n_detected_sources_listed" in df:
        mism = df["n_detected_sources_listed"] != df["n_detected"]
        if mism.any():
            warnings.append(f"{name}: {int(mism.sum())} runs list a different number of sources than n_detected")
    if "survey_duration_years" in df:
        d = df["survey_duration_years"].astype(float)
        if ((d < 1.0) | (d > 20.0)).any():
            errors.append(f"{name}: implausible survey durations {sorted(d.unique())[:5]}")
    # VT should be proportional to efficiency at fixed z_max (same injection volume)
    # survey_sim weights each recovery individually, so VT/efficiency fluctuates from run to run by a few
    # per cent; a large spread within one model at fixed z_max indicates a normalisation problem.
    if {"z_max", "efficiency", "model"} <= set(df.columns):
        for (m, zm), g in df.groupby(["model", "z_max"]):
            g = g[g["n_detected"] >= 100]
            if len(g) >= 2:
                k = g["vt_gpc3_yr"] / g["efficiency"]
                spread = (k.max() - k.min()) / k.median()
                if spread > 0.15:
                    warnings.append(f"{name}: VT/efficiency for {m} varies by {spread:.1%} at z_max={zm} "
                                    f"(check normalisation)")
    if "seed" in df and df["seed"].notna().any():
        dup = df["seed"].dropna().duplicated(keep=False)
        if dup.any():
            warnings.append(f"{name}: {int(dup.sum())} runs share a random seed (realisations may be correlated)")
    # Duplicate realisations (re-runs of the same task): keep the newest file
    keys = [c for c in ("model", "task_id", "run_index") if c in df.columns]
    if keys and "file_timestamp" in df:
        d = df.sort_values("file_timestamp").duplicated(subset=keys, keep="last")
        if d.any():
            warnings.append(f"{name}: dropped {int(d.sum())} older duplicate realisations of the same task")
            df = df.sort_values("file_timestamp")[~d.values].sort_index()
    return df


def validate_tables(tables: Dict[str, pd.DataFrame], previous=None) -> Tuple[Dict[str, pd.DataFrame], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    for name, cols in REQUIRED_COLUMNS.items():
        if name in tables:
            missing = [c for c in cols if c not in tables[name].columns]
            if missing:
                errors.append(f"{name}: missing columns {missing}")
    for name in list(tables):
        if name.endswith("_runs") and name in REQUIRED_COLUMNS and not any(e.startswith(name) for e in errors):
            if "efficiency" in tables[name].columns or "n_detected" in tables[name].columns:
                tables[name] = _check_runs(name, tables[name], errors, warnings)
    if "suite02_fiducial" in tables:
        f = tables["suite02_fiducial"]
        counts = f.groupby("model").size()
        if (counts != 1).any():
            errors.append(f"suite02_fiducial: expected exactly one fiducial row per model, got {counts.to_dict()}")
    # Regression guard: the number of realisations per model must not shrink
    if previous is not None:
        for name in ("suite01_runs", "suite07_runs", "suite03_runs"):
            if name in previous.tables and name in tables:
                key = "model" if "model" in tables[name] else "population_type"
                old = previous.tables[name].groupby(key).size()
                new = tables[name].groupby(key).size()
                for m, n_old in old.items():
                    n_new = int(new.get(m, 0))
                    if n_new == 0 and m in RETIRED_POPULATIONS.get(name, ()):
                        warnings.append(f"{name}: retired population {m} dropped ({n_old} runs in previous snapshot)")
                        continue
                    if n_new < n_old:
                        errors.append(f"{name}: {m} has {n_new} runs, previous snapshot had {n_old} "
                                      f"(refusing to replace good data with a partial set; use --allow-shrink)")
            elif name in previous.tables and name not in tables:
                errors.append(f"{name}: present in previous snapshot but missing now (results/ not mounted?)")
    for name, df in tables.items():
        if df is None or df.empty:
            warnings.append(f"{name}: empty table")
    return tables, errors, warnings


# ---------------------------------------------------------------------------
# Writing / verification
# ---------------------------------------------------------------------------
def _git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                   timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def _table_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format="%.10g")
    return buf.getvalue().encode()


def write_snapshot(tables: Dict[str, pd.DataFrame], manifest: dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payloads = {f"{name}.csv": _table_bytes(df) for name, df in sorted(tables.items())}
    manifest["tables"] = {
        name[:-4]: {"rows": int(len(tables[name[:-4]])), "columns": list(tables[name[:-4]].columns),
                    "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in payloads.items()
    }
    fd, tmp = tempfile.mkstemp(prefix=".tmp_snapshot_", suffix=".zip", dir=dest.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for name, data in payloads.items():
                zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))  # reproducible archive
                zi.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zi, data)
            zi = zipfile.ZipInfo("manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, json.dumps(manifest, indent=2, sort_keys=True, default=str))
        # verify by reading back before replacing anything
        with zipfile.ZipFile(tmp) as zf:
            if zf.testzip() is not None:
                raise RuntimeError("zip integrity check failed")
            man = json.loads(zf.read("manifest.json"))
            for t, info in man["tables"].items():
                data = zf.read(f"{t}.csv")
                if hashlib.sha256(data).hexdigest() != info["sha256"]:
                    raise RuntimeError(f"checksum mismatch for {t}")
                if len(pd.read_csv(io.BytesIO(data))) != info["rows"] and info["rows"] > 0:
                    raise RuntimeError(f"row-count mismatch for {t}")
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return dest


def build(results_dir: Path = DEFAULT_RESULTS, previous=None, force: bool = False,
          max_source_runs: int = 10, allow_shrink: bool = False, only: Optional[Sequence[str]] = None):
    """Run all collectors and validation. Returns (tables, manifest, errors)."""
    ctx = Context(results_dir=Path(results_dir), previous=previous, force=force,
                  max_source_runs=max_source_runs)
    tables: Dict[str, pd.DataFrame] = {}
    t0 = time.time()
    for col in REGISTRY:
        if only and col.name not in only:
            print(f"[collect] {col.name}: skipped (--suites); previous tables carried forward", flush=True)
            if previous is not None:
                for t, df in previous.tables.items():
                    if t.startswith(f"{col.name}_"):
                        tables[t] = df
                if col.name in previous.manifest.get("fingerprints", {}):
                    ctx.fingerprints[col.name] = previous.manifest["fingerprints"][col.name]
                ctx.reused.append(col.name)
            continue
        print(f"[collect] {col.name}", flush=True)
        try:
            out = col.func(ctx)
        except Exception as exc:  # a broken collector must not produce a partial snapshot silently
            ctx.warn(f"collector {col.name} raised {type(exc).__name__}: {exc}")
            raise
        for k, v in (out or {}).items():
            tables[f"{col.name}_{k}"] = v
    prev_for_guard = None if allow_shrink else previous
    tables, errors, warnings = validate_tables(tables, prev_for_guard)
    # carry forward suites that exist in the previous snapshot but whose raw files are absent here
    if previous is not None:
        for name, df in previous.tables.items():
            if name not in tables:
                tables[name] = df
                warnings.append(f"{name}: raw results not found locally; carried forward from previous snapshot")
        for k, v in previous.manifest.get("fingerprints", {}).items():
            ctx.fingerprints.setdefault(k, v)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime(STAMP_FMT),
        "host": socket.gethostname(),
        "results_dir": str(results_dir),
        "git": _git_info(),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "fingerprints": ctx.fingerprints,
        "reused_from_previous": ctx.reused,
        "skipped_files": ctx.skipped_files,
        "warnings": ctx.warnings + warnings,
        "errors": errors,
        "build_seconds": round(time.time() - t0, 1),
        "notes": ("suite05 and suite06 result files under results/ are NOT ingested: they are generated from "
                  "hard-coded efficiencies (binomial re-draws), not from survey_sim simulations. Convergence and "
                  "inclination analyses are derived from suite01/suite07 realisations instead."),
    }
    return tables, manifest, errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .snapshot import load_snapshot  # local import to avoid cycles

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--out-dir", default=str(SNAPSHOT_DIR))
    ap.add_argument("--timestamped", action="store_true",
                    help="write snapshots/knrates_results_<UTC timestamp>.zip instead of the 'latest' file")
    ap.add_argument("--also-latest", action="store_true", help="with --timestamped, also replace the latest file")
    ap.add_argument("--dry-run", action="store_true", help="build and validate only")
    ap.add_argument("--force", action="store_true", help="re-read all suites (ignore cache)")
    ap.add_argument("--allow-shrink", action="store_true",
                    help="permit a suite to have fewer runs than the previous snapshot")
    ap.add_argument("--suites", nargs="+", default=None,
                    help="only (re)collect these suites (e.g. suite03 models); others are carried forward")
    ap.add_argument("--max-source-runs", type=int, default=10,
                    help="runs per model whose per-source rows are stored (inclination models: all runs)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / LATEST_NAME
    # serialise concurrent updates (e.g. two aggregation jobs finishing together)
    import fcntl
    lock_fh = open(out_dir / ".collect.lock", "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
    except OSError as exc:  # some network file systems do not support flock
        print(f"[collect] warning: could not lock {out_dir}/.collect.lock ({exc}); continuing")
    previous = None
    if latest.exists():
        try:
            previous = load_snapshot(latest)
        except Exception as exc:
            print(f"[collect] could not read previous snapshot ({exc}); building from scratch")
    if not Path(args.results_dir).exists():
        print(f"[collect] results directory {args.results_dir} not found; nothing to do")
        return 1
    tables, manifest, errors = build(Path(args.results_dir), previous, args.force,
                                     args.max_source_runs, args.allow_shrink, args.suites)
    print(f"[collect] {len(tables)} tables, {len(manifest['warnings'])} warnings, {len(errors)} errors, "
          f"{len(manifest['skipped_files'])} skipped files")
    for w in manifest["warnings"]:
        print(f"  [warning] {w}")
    for e in errors:
        print(f"  [ERROR] {e}")
    if errors:
        print("[collect] validation failed -> no snapshot written; existing snapshots untouched")
        return 1
    if args.dry_run:
        print("[collect] dry run: nothing written")
        return 0
    targets = []
    if args.timestamped:
        targets.append(out_dir / f"knrates_results_{manifest['created_utc']}.zip")
        if args.also_latest:
            targets.append(latest)
    else:
        targets.append(latest)
    for t in targets:
        if t != latest and t.exists():
            print(f"[collect] {t} exists; refusing to overwrite a time-stamped snapshot")
            return 1
        write_snapshot(tables, dict(manifest), t)
        print(f"[collect] wrote {t} ({t.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

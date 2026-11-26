#!/usr/bin/env python3
"""Suite 02 high-performance 'Filter-Many' criteria sweep engine.

Purpose & Architecture
----------------------
Implements the second stage of the Simulate-Once / Filter-Many architecture for Suite 02:
1. ``simulate_candidates.py`` injects synthetic kilonovae once under a loose bounding envelope
   (``min_galactic_lat = 15.0`` deg, ``z_max = 0.15``) and saves all loose-passing light curves.
2. **This module** loads those candidate light curves into :class:`CandidatePool`, precomputes
   per-band weighted least-squares (WLS) rates via :func:`knrates.criteria.band_wls_rates` at
   each secondary S/N threshold, and evaluates tens of thousands of 10-parameter criteria
   combinations in seconds without re-running ``survey_sim``.

Exact Parity Guarantees
-----------------------
- Uses :func:`knrates.criteria.band_wls_rates` and :func:`knrates.criteria.passes` directly.
- Enforces ``survey_sim`` span semantics: ``t_last - t_first <= max_timespan_days`` over all
  qualifying detections (no rolling sub-windows).
- Enforces ``survey_sim`` fast-transient OR semantics: passes if ``max_fade >= min_fade_rate``
  OR ``max_rise >= min_rise_rate`` (when the corresponding threshold is ``> 0``).
- Rejects grid points whose ``min_galactic_lat`` differs from the loose simulation cut when
  per-source sky coordinates are absent from the candidate JSON.

Inputs & Outputs
----------------
- **Input:** ``results/02_granular_criteria_sweep/candidates/<Model>/candidates_*.json``
  (plus ``.metadata.json`` sidecars).
- **Outputs (``--output-dir``):**
  ``granular_sweep_metrics_<Model>.csv``, ``granular_sweep_metrics.csv``,
  ``criteria_sensitivity_ranking_<Model>.csv``, and ``criteria_sensitivity_ranking.csv``.

CLI Examples
------------
    python analysis_suites/02_granular_criteria_sweep/fast_criteria_filter.py --grid-preset ultra_fine
    python analysis_suites/02_granular_criteria_sweep/fast_criteria_filter.py --merge-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

_SUITE_DIR = str(Path(__file__).resolve().parent)
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
for _p in (_SUITE_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from knrates.criteria import band_wls_rates, passes as criteria_passes  # noqa: E402
from run_granular_sweep import get_grid_combinations  # noqa: E402

MODELS = ["Bu2026Fixed", "Bu2026Vary", "Metzger"]
PRIM_THRESHOLDS = (3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
SEC_THRESHOLDS = (2.0, 2.5, 3.0, 3.5, 4.0)
LATS = (0.0, 5.0, 10.0, 15.0, 20.0)
PRIM_MAP = {pt: idx for idx, pt in enumerate(PRIM_THRESHOLDS)}

CRITERIA_COLS = [
    "snr_threshold", "snr_threshold_secondary", "min_detections",
    "min_detections_primary", "max_timespan_days", "min_time_separation_hours",
    "min_rise_rate", "min_fade_rate", "min_galactic_lat", "require_fast_transient",
]

FILTER_VERSION = "2026-11b"
REQUIRED_LOOSE_GAL_LAT = 15.0
REQUIRED_Z_MAX = 0.15


def total_survey_vt_gpc3_yr(z_max: float) -> float:
    """Fallback analytic survey volume-time [Gpc^3 yr] when loose-run ``rate_summaries`` is absent."""
    dh = 4.42825
    dc = dh * (z_max - 0.2675 * (z_max ** 2) + 0.145 * (z_max ** 3))
    vc = (4.0 / 3.0) * math.pi * (dc ** 3)
    return vc * 0.6687 * 8.0057


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Suite 02 criteria filter."""
    ap = argparse.ArgumentParser(
        description="Filter candidate light curves across the Suite 02 criteria sweep grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--candidates-dir", default="results/02_granular_criteria_sweep/candidates",
                    help="Directory containing candidate JSON result files.")
    ap.add_argument("--output-dir", default="results/02_granular_criteria_sweep",
                    help="Directory to save sweep metrics CSVs and sensitivity rankings.")
    ap.add_argument("--grid-preset", default="ultra_fine",
                    choices=["ultra_fine", "broad", "fine", "legacy_ultra_fine", "canonical", "all_slices"],
                    help="Grid configuration preset to evaluate.")
    ap.add_argument("--model", default="all", choices=["all", *MODELS],
                    help="Kilonova model(s) to evaluate ('all' processes all three).")
    ap.add_argument("--mode", default="auto", choices=["auto", "fiducial", "ensemble"],
                    help="Processing mode ('auto' uses ensemble mode when >1 sample file exists).")
    ap.add_argument("--n-workers", type=int, default=max(1, os.cpu_count() or 1),
                    help="Parallel worker processes for grid chunk evaluation.")
    ap.add_argument("--export-compact-jsons", action="store_true",
                    help="Retained for CLI compatibility.")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Optional cap on candidate sample files evaluated per model.")
    ap.add_argument("--merge-only", action="store_true",
                    help="Merge existing per-model sweep CSVs without re-filtering candidates.")
    return ap.parse_args()


class CandidatePool:
    """Pre-indexed in-memory cache of loose-envelope candidate light curves for rapid grid sweeps."""

    def __init__(self, filepath: Path, model_name: str):
        self.filepath = Path(filepath)
        self.model_name = model_name
        self.data = json.loads(self.filepath.read_text())
        self.n_simulated = int(self.data.get("n_simulated", 1_000_000))

        meta_p = Path(f"{self.filepath}.metadata.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        self.z_max = float((meta.get("params") or {}).get("z_max", 0.15 if model_name in {"Bu2026Fixed", "Bu2026Vary"} else 0.30))

        rs = (self.data.get("rate_summaries") or [{}])[0]
        eff_loose = float(rs.get("n_detected", 0)) / max(float(rs.get("n_simulated", 1)), 1.0)
        if eff_loose > 0 and float(rs.get("effective_vt_gpc3_yr", 0)) > 0:
            self.vt_total = float(rs["effective_vt_gpc3_yr"]) / eff_loose
        else:
            self.vt_total = total_survey_vt_gpc3_yr(self.z_max)
            print(f"  [warn] {self.filepath.name}: no loose-run VT; falling back to analytic normalisation", flush=True)

        self.loose_min_galactic_lat = float((meta.get("loose_criteria") or {}).get("min_galactic_lat", 0.0))
        sources = self.data.get("detected_sources") or []
        self.has_positions = any(("galactic_lat" in x) or ("gal_lat" in x) for x in sources[:50])
        self.n_candidates = len(sources)
        self.lightcurves: List[Dict[str, Any]] = []

        for src in sources:
            p_times = [float(t) for t in src.get("photometry_times", []) if math.isfinite(float(t))]
            p_mags = [float(m) for m in src.get("photometry_mags", []) if math.isfinite(float(m))]
            p_errs = [float(e) for e in src.get("photometry_errs", []) if math.isfinite(float(e))]
            p_snrs = [float(s) for s in src.get("photometry_snrs", []) if math.isfinite(float(s))]
            p_bands = [str(b) for b in src.get("photometry_bands", [])]
            if not p_snrs and p_errs:
                p_snrs = [(1.08574 / e) if e > 0 else 0.0 for e in p_errs]
            if not p_times or not p_snrs:
                continue

            gal_lat = float(src.get("galactic_lat", src.get("gal_lat", 90.0)))
            gal_lat = gal_lat if math.isfinite(gal_lat) else 90.0

            times_arr = np.asarray(p_times, dtype=np.float64)
            snrs_arr = np.asarray(p_snrs, dtype=np.float64)
            mags_arr = np.asarray(p_mags, dtype=np.float64) if p_mags else np.zeros_like(times_arr)
            bands_arr = np.asarray(p_bands if len(p_bands) == len(times_arr) else ["x"] * len(times_arr))

            order = np.argsort(times_arr)
            times_arr, snrs_arr, mags_arr, bands_arr = times_arr[order], snrs_arr[order], mags_arr[order], bands_arr[order]
            times_list = times_arr.tolist()
            snrs_list = snrs_arr.tolist()
            mags_list = mags_arr.tolist()
            bands_list = [str(x) for x in bands_arr]

            qual_by_sec: Dict[float, Any] = {}
            for sec in SEC_THRESHOLDS:
                sec_th = sec - 1e-4
                pts = [
                    (times_list[k], snrs_list[k], mags_list[k], bands_list[k])
                    for k in range(len(times_list)) if snrs_list[k] >= sec_th
                ]
                n_q = len(pts)
                if n_q == 0:
                    qual_by_sec[sec] = None
                elif n_q == 1:
                    qual_by_sec[sec] = (1, pts[0][1])
                else:
                    prim_counts = tuple(sum(1 for p in pts if p[1] >= (pt - 1e-4)) for pt in PRIM_THRESHOLDS)
                    t_span = pts[-1][0] - pts[0][0]
                    mfade, mrise = band_wls_rates(pts)
                    qual_by_sec[sec] = (n_q, prim_counts, t_span, t_span * 24.0, mfade, mrise)

            self.lightcurves.append({
                "times": times_arr,
                "snrs": snrs_arr,
                "mags": mags_arr,
                "bands": bands_list,
                "abs_gal_lat": abs(gal_lat),
                "max_snr": max(snrs_list),
                "n_pts": len(times_arr),
                "qual_by_sec": qual_by_sec,
            })

        self.index_by_sec_lat: Dict[Tuple[float, float], List[Tuple[float, Any]]] = {}
        for sec in SEC_THRESHOLDS:
            for lat in LATS:
                lat_th = lat - 1e-4
                self.index_by_sec_lat[(sec, lat)] = [
                    (c["max_snr"], c["qual_by_sec"][sec])
                    for c in self.lightcurves
                    if c["abs_gal_lat"] >= lat_th and c["qual_by_sec"].get(sec) is not None
                ]


def evaluate_candidate_against_criteria(lc: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """Evaluate one candidate dict ``lc`` against ``cfg`` using canonical :func:`knrates.criteria.passes`."""
    if lc["abs_gal_lat"] < (cfg["min_galactic_lat"] - 1e-4):
        return False
    times = lc["times"]
    snrs = lc["snrs"]
    mags = lc["mags"]
    bands = lc.get("bands") or (["r"] * len(times))
    pts = [(float(t), float(s), float(m), str(b)) for t, s, m, b in zip(times, snrs, mags, bands)]
    return criteria_passes(pts, cfg)


def evaluate_grid_chunk_counts(chunk_cfgs: List[Dict[str, Any]], pool: CandidatePool) -> np.ndarray:
    """Evaluate a slice of criteria grid configurations against ``pool``'s precomputed indices."""
    for cfg in chunk_cfgs:
        if not pool.has_positions and abs(cfg["min_galactic_lat"] - pool.loose_min_galactic_lat) > 1e-6:
            raise ValueError(
                f"grid point |b|>={cfg['min_galactic_lat']} differs from the |b| cut applied when the candidates were "
                f"simulated (|b|>={pool.loose_min_galactic_lat}); positions are not stored, so it cannot be evaluated"
            )

    counts = np.zeros(len(chunk_cfgs), dtype=np.int32)
    index = pool.index_by_sec_lat

    for idx, cfg in enumerate(chunk_cfgs):
        c_list = index.get((cfg["snr_threshold_secondary"], cfg["min_galactic_lat"]))
        if not c_list:
            continue
        p_idx = PRIM_MAP.get(cfg["snr_threshold"], 0)
        p_th = cfg["snr_threshold"] - 1e-4
        m_det = cfg["min_detections"]
        m_prim = cfg["min_detections_primary"]
        m_span = cfg["max_timespan_days"] + 1e-4
        m_sep = cfg["min_time_separation_hours"] - 1e-4
        m_fade = (cfg.get("min_fade_rate", 0.0) - 1e-4) if cfg.get("min_fade_rate", 0.0) > 0 else 0.0
        m_rise = (cfg.get("min_rise_rate", 0.0) - 1e-4) if cfg.get("min_rise_rate", 0.0) > 0 else 0.0
        req_f = cfg.get("require_fast_transient", True)

        cnt = 0
        for max_s, data in c_list:
            if m_prim >= 1 and max_s < p_th:
                continue
            if data[0] == 1:
                if m_det == 1 and m_prim <= 1 and (m_prim == 0 or data[1] >= p_th):
                    cnt += 1
                continue
            n_q, prim_counts, t_span, t_sep_hrs, mfade, mrise = data
            if n_q < m_det or prim_counts[p_idx] < m_prim or t_span > m_span or t_sep_hrs < m_sep:
                continue
            if req_f and (m_fade > 0.0 or m_rise > 0.0):
                if not ((m_fade > 0.0 and mfade >= m_fade) or (m_rise > 0.0 and mrise >= m_rise)):
                    continue
            cnt += 1
        counts[idx] = cnt
    return counts


def run_filter_counts_for_pool(pool: CandidatePool, grid: List[Dict[str, Any]], n_workers: int = 4) -> np.ndarray:
    """Evaluate ``grid`` against ``pool`` and return a 1D ``int32`` detection-count array."""
    total_pts = len(grid)
    if n_workers <= 1 or total_pts < 50:
        return evaluate_grid_chunk_counts(grid, pool)
    chunk_size = math.ceil(total_pts / n_workers)
    chunks = [grid[i : i + chunk_size] for i in range(0, total_pts, chunk_size)]
    with mp.Pool(processes=min(n_workers, len(chunks))) as p:
        return np.concatenate(p.starmap(evaluate_grid_chunk_counts, [(c, pool) for c in chunks]))


def run_filter_for_pool(pool: CandidatePool, grid: List[Dict[str, Any]], n_workers: int = 4) -> pd.DataFrame:
    """Evaluate ``grid`` against ``pool`` and return a summary DataFrame."""
    counts = run_filter_counts_for_pool(pool, grid, n_workers=n_workers)
    eff = counts.astype(np.float64) / max(1, pool.n_simulated)
    vt_eff = eff * pool.vt_total
    with np.errstate(divide="ignore", invalid="ignore"):
        r95 = np.where(vt_eff > 0, -np.log(0.05) / vt_eff, np.nan)
    df = pd.DataFrame(grid)
    return df.assign(
        model=pool.model_name,
        n_detected=counts.astype(int),
        n_simulated=pool.n_simulated,
        overall_efficiency=eff,
        effective_vt_gpc3_yr=vt_eff,
        rate_upper_95=r95,
    )


def build_model_metrics_df(
    grid: List[Dict[str, Any]],
    counts_matrix: np.ndarray,
    pool: Any,
    cand_files: List[Path],
    vt_per_eff: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Aggregate an ``(n_samples, n_grid)`` detection-count matrix into per-grid-point ensemble statistics."""
    n_samples = len(cand_files)
    n_sim = float(pool.n_simulated)
    vt_norm = np.full(n_samples, float(pool.vt_total)) if vt_per_eff is None else np.asarray(vt_per_eff, dtype=np.float64)
    eff_matrix = counts_matrix.astype(np.float64) / max(1.0, n_sim)
    vt_matrix = eff_matrix * vt_norm.reshape(-1, 1)

    mean_counts = np.mean(counts_matrix, axis=0)
    mean_eff = np.mean(eff_matrix, axis=0)
    std_eff = np.std(eff_matrix, axis=0, ddof=1) if n_samples > 1 else np.zeros_like(mean_eff)
    mean_vt = np.mean(vt_matrix, axis=0)
    std_vt = np.std(vt_matrix, axis=0, ddof=1) if n_samples > 1 else np.zeros_like(mean_vt)
    root_n = math.sqrt(n_samples) if n_samples > 1 else 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_r95 = np.where(mean_vt > 0, -np.log(0.05) / mean_vt, np.nan)

    df = pd.DataFrame(grid)
    df["model"] = pool.model_name
    df["n_detected"] = mean_counts
    df["n_simulated"] = int(n_sim)
    df["overall_efficiency"] = mean_eff
    df["overall_efficiency_mean"] = mean_eff
    df["overall_efficiency_std"] = std_eff
    df["efficiency_sem"] = std_eff / root_n if n_samples > 1 else np.zeros_like(mean_eff)
    df["effective_vt_gpc3_yr"] = mean_vt
    df["effective_vt_gpc3_yr_mean"] = mean_vt
    df["effective_vt_gpc3_yr_std"] = std_vt
    df["vt_sem"] = std_vt / root_n if n_samples > 1 else np.zeros_like(mean_vt)
    df["rate_upper_95"] = mean_r95
    df["n_samples"] = n_samples
    return df


def merge_model_csvs(out_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Merge per-model sweep and ranking CSVs in ``out_dir`` into unified multi-model CSVs."""
    model_dfs, rank_dfs = [], []
    for m in MODELS:
        m_csv = out_dir / f"granular_sweep_metrics_{m}.csv"
        r_csv = out_dir / f"criteria_sensitivity_ranking_{m}.csv"
        if m_csv.exists():
            df_m = pd.read_csv(m_csv)
            model_dfs.append(df_m)
            print(f"  Found {m_csv.name} ({len(df_m):,} rows)", flush=True)
        if r_csv.exists():
            rank_dfs.append(pd.read_csv(r_csv))

    if not model_dfs:
        return None, None
    combined_metrics = pd.concat(model_dfs, ignore_index=True)
    combined_metrics.to_csv(out_dir / "granular_sweep_metrics.csv", index=False)
    combined_rank = pd.concat(rank_dfs, ignore_index=True) if rank_dfs else None
    if combined_rank is not None:
        combined_rank.to_csv(out_dir / "criteria_sensitivity_ranking.csv", index=False)
    return combined_metrics, combined_rank


def compute_criteria_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank the 10 selection criteria by marginal relative variation in detection efficiency."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for col in CRITERIA_COLS:
        if col not in df.columns:
            continue
        means = df.groupby(col)["overall_efficiency"].mean()
        if means.empty:
            continue
        min_eff, max_eff, base_eff = float(means.min()), float(means.max()), float(means.mean())
        span = max_eff - min_eff
        rows.append({
            "criterion": col,
            "mean_efficiency": base_eff,
            "min_efficiency": min_eff,
            "max_efficiency": max_eff,
            "efficiency_range": span,
            "relative_variation_pct": (span / base_eff * 100.0) if base_eff > 0 else 0.0,
            "n_configurations": len(df),
        })
    out = pd.DataFrame(rows).sort_values("relative_variation_pct", ascending=False).reset_index(drop=True)
    if not out.empty:
        out["impact_rank"] = np.arange(1, len(out) + 1)
    return out


def _candidate_meta(f: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(f"{f}.metadata.json").read_text())
    except Exception:
        return {}


def filter_compatible_candidates(files: List[Path]) -> Tuple[List[Path], List[str]]:
    """Keep deduplicated candidate files matching ``REQUIRED_LOOSE_GAL_LAT`` and ``REQUIRED_Z_MAX``."""
    keep, skipped = [], []
    for f in files:
        m = _candidate_meta(f)
        loose_b = float((m.get("loose_criteria") or {}).get("min_galactic_lat", -1))
        zmax = float((m.get("params") or {}).get("z_max", -1))
        if abs(loose_b - REQUIRED_LOOSE_GAL_LAT) > 1e-6 or abs(zmax - REQUIRED_Z_MAX) > 1e-6:
            skipped.append(f"{f.name} (loose |b|>={loose_b:g}, z_max={zmax:g})")
        else:
            keep.append(f)

    seen_seeds: Dict[Any, Path] = {}
    dedup = []
    for f in sorted(keep, key=lambda x: ("fiducial" in x.name, x.name)):
        sd = _candidate_meta(f).get("seed")
        if sd is not None and sd in seen_seeds:
            skipped.append(f"{f.name} (duplicate seed {sd} of {seen_seeds[sd].name})")
            continue
        seen_seeds[sd] = f
        dedup.append(f)
    return dedup, skipped


def vt_per_efficiency(f: Path) -> float:
    """Return ``survey_sim`` ``<VT> / efficiency`` normalization for candidate set ``f``."""
    m = _candidate_meta(f)
    eff, vt = float(m.get("overall_efficiency", 0) or 0), float(m.get("effective_vt_gpc3_yr", 0) or 0)
    return (vt / eff) if (eff > 0 and vt > 0) else float(CandidatePool(f, "unknown").vt_total)


def find_candidate_files(cand_dir: Union[str, Path], model_name: str) -> List[Path]:
    """Locate candidate JSON files (excluding ``.metadata.json``) across nested and flat directory layouts."""
    base = Path(cand_dir)
    search_dirs = [base / model_name, base / "candidates" / model_name, base, base / "candidates"]
    if base.parent != base:
        search_dirs.extend([base.parent / model_name, base.parent])

    seen = set()
    for d in search_dirs:
        if not d.is_dir():
            continue
        patterns = [f"candidates_{model_name}_*.json", f"candidates_{model_name}.json", f"*{model_name}*.json"]
        if d.name.lower() == model_name.lower():
            patterns.insert(0, "candidates_*.json")
        matched = [
            p for pat in patterns for p in sorted(d.glob(pat))
            if not p.name.endswith(".metadata.json") and not (p in seen or seen.add(p))
        ]
        if matched:
            return matched
    return []


def main() -> None:
    args = parse_args()
    cand_dir, out_dir = Path(args.candidates_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        m_df, _ = merge_model_csvs(out_dir)
        if m_df is None:
            sys.exit(f"Error: No per-model CSVs found to merge in {out_dir}")
        return

    grid = get_grid_combinations(args.grid_preset)
    target_models = MODELS if args.model == "all" else [args.model]
    generated = 0

    for model_name in target_models:
        cand_files = find_candidate_files(cand_dir, model_name)
        if not cand_files:
            print(f"[SKIP] No candidate files found for {model_name} under {cand_dir}", flush=True)
            continue
        cand_files, skipped = filter_compatible_candidates(cand_files)
        if skipped:
            print(f"  [skip] {len(skipped)} incompatible/duplicate set(s); e.g. {skipped[0]}", flush=True)
        if not cand_files:
            continue
        if args.max_samples and args.max_samples > 0:
            cand_files = cand_files[:args.max_samples]

        fid_file = next((f for f in cand_files if "fiducial" in f.name), None)
        if fid_file is None and args.mode == "fiducial" and len(cand_files) == 1:
            fid_file = cand_files[0]
        is_ensemble = (len(cand_files) > 1) or (args.mode == "ensemble") or (fid_file is None)

        if not is_ensemble and fid_file is not None and fid_file.exists():
            t0 = time.perf_counter()
            pool = CandidatePool(fid_file, model_name)
            counts = run_filter_counts_for_pool(pool, grid, n_workers=args.n_workers)
            df_model = build_model_metrics_df(grid, counts.reshape(1, -1), pool, [fid_file], np.array([pool.vt_total]))
            print(f"  [{model_name}] evaluated {len(grid):,} points in {time.perf_counter() - t0:.2f}s", flush=True)
        else:
            key = hashlib.sha1(("|".join(f.name for f in cand_files) + FILTER_VERSION + str(len(grid))).encode()).hexdigest()[:12]
            chk_file = out_dir / f".checkpoint_{model_name}_{len(cand_files)}samples_{key}.npy"
            counts_matrix = np.zeros((len(cand_files), len(grid)), dtype=np.int32)
            if chk_file.exists():
                try:
                    counts_matrix = np.load(chk_file)
                except Exception:
                    pass
            first_pool = None
            for s_idx, f in enumerate(cand_files):
                if s_idx < len(counts_matrix) and np.any(counts_matrix[s_idx] > 0):
                    continue
                t0 = time.perf_counter()
                pool = CandidatePool(f, model_name)
                first_pool = first_pool or pool
                counts_matrix[s_idx] = run_filter_counts_for_pool(pool, grid, n_workers=args.n_workers)
                print(f"  [{model_name} {s_idx + 1}/{len(cand_files)}] {f.name} in {time.perf_counter() - t0:.2f}s", flush=True)
                try:
                    np.save(chk_file, counts_matrix)
                except Exception:
                    pass
            first_pool = first_pool or CandidatePool(cand_files[0], model_name)
            df_model = build_model_metrics_df(grid, counts_matrix, first_pool, cand_files,
                                              np.array([vt_per_efficiency(f) for f in cand_files]))
            if chk_file.exists():
                chk_file.unlink(missing_ok=True)

        rank_df = compute_criteria_ranking(df_model).assign(model=model_name)
        df_model.to_csv(out_dir / f"granular_sweep_metrics_{model_name}.csv", index=False)
        rank_df.to_csv(out_dir / f"criteria_sensitivity_ranking_{model_name}.csv", index=False)
        generated += 1

    if generated == 0:
        sys.exit(1)
    merge_model_csvs(out_dir)


if __name__ == "__main__":
    main()

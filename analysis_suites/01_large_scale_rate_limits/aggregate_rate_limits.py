#!/usr/bin/env python3
"""Aggregate Suite 01 kilonova simulation outputs into summary CSVs.

Purpose
-------
Parses serialized ``survey_sim`` JSON result files across Suite 01 models
(``Bu2026Fixed``, ``Bu2026Vary``, ``Metzger``) and computes per-realisation Poisson
volumetric rate upper limits (``R_90``, ``R_95``, ``R_99``) alongside legacy
duration-scaled comparison columns.

Note for Researchers
--------------------
For publication headline numbers, use :func:`knrates.analysis.headline_limits` and
epoch-resolved :func:`knrates.analysis.andreoni_comparison` on the compiled snapshot
(``snapshots/knrates_results_latest.zip``), which pool realisations by injection-weighted
``<VT>`` rather than median-of-run limits.

CLI Example
-----------
    python analysis_suites/01_large_scale_rate_limits/aggregate_rate_limits.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ANDREONI_FIXED_95 = 1775.0
ANDREONI_ISOTROPIC_95 = 4029.0
ANDREONI_DURATION_YRS = 23.0 / 12.0

_ALIAS_PAIRS = (
    ("rate_90_upper_limit", "rate_90_zero_events"),
    ("rate_95_upper_limit", "rate_95_zero_events"),
    ("rate_99_upper_limit", "rate_99_zero_events"),
    ("rate_95_duration_scaled_1_92yr", "duration_scaled_95"),
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate Suite 01 kilonova rate limits from simulation JSONs.")
    ap.add_argument("--glob", dest="pattern", default="results/01_large_scale_rate_limits/**/*.json",
                    help="Glob pattern for result JSON files.")
    ap.add_argument("--output-csv", default="results/01_large_scale_rate_limits/rate_limit_summary.csv",
                    help="Path to write the aggregated summary CSV.")
    ap.add_argument("--include-legacy", action="store_true",
                    help="Include legacy marathon results under results/marathon/.")
    return ap.parse_args()


def extract_model_from_path(file_path: str) -> str:
    """Infer canonical Suite 01 model name from ``file_path``."""
    low = file_path.lower()
    if "metzger" in low:
        return "Metzger"
    if "bu2026fixed" in low or "ztf_fixed" in low or "bu2026_fixed" in low:
        return "Bu2026Fixed"
    if "bu2026vary" in low or "bu2026_vary" in low:
        return "Bu2026Vary"
    return "Unknown"


def process_file(path: Path) -> List[Dict[str, Any]]:
    """Extract rate summary rows from a single ``survey_sim`` JSON file."""
    if path.name.endswith(".metadata.json"):
        return []
    try:
        content = json.loads(path.read_text())
    except Exception:
        return []

    meta_path = Path(f"{path}.metadata.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    model = meta.get("model", extract_model_from_path(str(path)))
    rate_summaries = content.get("rate_summaries", []) if isinstance(content, dict) else (content if isinstance(content, list) else [])

    rows: List[Dict[str, Any]] = []
    for rs in rate_summaries:
        eff_vt = float(rs.get("effective_vt_gpc3_yr", float("nan")))
        if not math.isfinite(eff_vt) or eff_vt <= 0:
            continue
        duration = float(rs.get("survey_duration_years", 8.0057))
        r90 = -math.log(0.10) / eff_vt
        r95 = -math.log(0.05) / eff_vt
        r99 = -math.log(0.01) / eff_vt
        r95_scaled = r95 * (duration / ANDREONI_DURATION_YRS)
        n_det = int(rs.get("n_detected", content.get("n_detected", 0)))

        proj_map = {
            "Bu2026Fixed": ANDREONI_FIXED_95 * (ANDREONI_DURATION_YRS / duration),
            "Bu2026Vary": ANDREONI_ISOTROPIC_95 * (ANDREONI_DURATION_YRS / duration),
        }
        rows.append({
            "file": str(path),
            "model": model,
            "transient_type": rs.get("transient_type", "Kilonova"),
            "n_simulated": rs.get("n_simulated", content.get("n_simulated", 1_000_000)),
            "n_detected": n_det,
            "overall_efficiency": rs.get("overall_efficiency", 0.0),
            "survey_duration_years": duration,
            "effective_vt_gpc3_yr": eff_vt,
            "rate_90_upper_limit": r90,
            "rate_95_upper_limit": r95,
            "rate_99_upper_limit": r99,
            "rate_90_zero_events": r90,
            "rate_95_zero_events": r95,
            "rate_99_zero_events": r99,
            "duration_scaled_95": r95_scaled,
            "rate_95_duration_scaled_1_92yr": r95_scaled,
            "andreoni_projected_8yr_r95": proj_map.get(model, float("nan")),
            "is_poisson_sparse": n_det < 5,
            "ratio_to_andreoni_fixed_1775": r95_scaled / ANDREONI_FIXED_95,
            "ratio_to_andreoni_isotropic_4029": r95_scaled / ANDREONI_ISOTROPIC_95,
        })
    return rows


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for f in sorted(glob.glob(args.pattern, recursive=True)):
        if "metadata" in f or (not args.include_legacy and "results/marathon" in f and "01_large_scale" in args.pattern):
            continue
        rows.extend(process_file(Path(f)))

    if not rows and Path("results/rate_limit_benchmark.csv").exists():
        df_bench = pd.read_csv("results/rate_limit_benchmark.csv")
        df_bench["model"] = df_bench["file"].apply(extract_model_from_path)
        rows = df_bench.to_dict(orient="records")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid simulation summaries found.")
        return

    for col_a, col_b in _ALIAS_PAIRS:
        if col_a not in df.columns and col_b in df.columns:
            df[col_a] = df[col_b]
        if col_b not in df.columns and col_a in df.columns:
            df[col_b] = df[col_a]
    if "is_poisson_sparse" not in df.columns:
        df["is_poisson_sparse"] = (df["n_detected"] < 5) if "n_detected" in df.columns else False

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} simulation records to {out_csv}")

    summary_table = df.groupby("model").agg(
        n_runs=("effective_vt_gpc3_yr", "count"),
        n_detected_mean=("n_detected", "mean"),
        efficiency_mean=("overall_efficiency", "mean"),
        vt_eff_mean=("effective_vt_gpc3_yr", "mean"),
        vt_eff_median=("effective_vt_gpc3_yr", "median"),
        r95_8yr_median=("rate_95_upper_limit", "median"),
        r95_scaled_median=("rate_95_duration_scaled_1_92yr", "median"),
        sparse_runs=("is_poisson_sparse", "sum"),
    )
    print("\n" + summary_table.to_string())


if __name__ == "__main__":
    main()

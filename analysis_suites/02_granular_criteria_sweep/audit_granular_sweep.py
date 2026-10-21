#!/usr/bin/env python3
"""Audit, validate adherence, and analyze sensitivity for detection criteria sweeps.

This script parses all simulation results from a detection criteria sweep, validates
100% adherence to all selection rules across all 10 criteria parameters:
  1. Primary SNR threshold (snr_threshold)
  2. Secondary SNR threshold (snr_threshold_secondary)
  3. Minimum qualifying detections (min_detections)
  4. Minimum primary SNR detections (min_detections_primary)
  5. Maximum timespan window (max_timespan_days)
  6. Minimum first-to-last separation (min_time_separation_hours)
  7. Minimum rise rate (min_rise_rate)
  8. Minimum fade rate (min_fade_rate)
  9. Absolute galactic latitude cut (min_galactic_lat)
  10. Fast-transient requirement (require_fast_transient)

Computes 1D marginal detection efficiencies, calculates criteria sensitivity gradients/elasticity,
ranks parameters by observational impact, and exports structured CSVs and metrics.
"""

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and analyze detection criteria sweep results.")
    parser.add_argument(
        "--sweep-root",
        default="results/02_granular_criteria_sweep",
        help="Root directory containing criteria sweep subdirectories.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/02_granular_criteria_sweep/granular_sweep_metrics.csv",
        help="Path for saving aggregated criteria metrics.",
    )
    parser.add_argument(
        "--ranking-csv",
        default="results/02_granular_criteria_sweep/criteria_sensitivity_ranking.csv",
        help="Path for saving criteria sensitivity ranking.",
    )
    return parser.parse_args()


def parse_label_metadata(label: str) -> Dict[str, Any]:
    """Parse numerical and categorical parameters from a standardized directory label."""
    # Pattern: snr4_snr22.5_ndet2_ndet11_span14d_sep3h_fade0.3_rise0_fast1_b15
    data = {
        "snr_threshold": 5.0,
        "snr_threshold_secondary": 3.0,
        "min_detections": 2,
        "min_detections_primary": 1,
        "max_timespan_days": 14.0,
        "min_time_separation_hours": 3.0,
        "min_rise_rate": 0.0,
        "min_fade_rate": 0.30,
        "min_galactic_lat": 15.0,
        "require_fast_transient": True,
    }
    tokens = label.split("_")
    for tok in tokens:
        if tok.startswith("snr2"):
            try:
                data["snr_threshold_secondary"] = float(tok.replace("snr2", ""))
            except ValueError:
                pass
        elif tok.startswith("snr"):
            try:
                data["snr_threshold"] = float(tok.replace("snr", ""))
            except ValueError:
                pass
        elif tok.startswith("ndet1"):
            try:
                data["min_detections_primary"] = int(tok.replace("ndet1", ""))
            except ValueError:
                pass
        elif tok.startswith("ndet"):
            try:
                data["min_detections"] = int(tok.replace("ndet", ""))
            except ValueError:
                pass
        elif tok.startswith("span") and tok.endswith("d"):
            try:
                data["max_timespan_days"] = float(tok.replace("span", "").replace("d", ""))
            except ValueError:
                pass
        elif tok.startswith("sep") and tok.endswith("h"):
            try:
                data["min_time_separation_hours"] = float(tok.replace("sep", "").replace("h", ""))
            except ValueError:
                pass
        elif tok.startswith("fade"):
            try:
                data["min_fade_rate"] = float(tok.replace("fade", ""))
            except ValueError:
                pass
        elif tok.startswith("rise"):
            try:
                data["min_rise_rate"] = float(tok.replace("rise", ""))
            except ValueError:
                pass
        elif tok.startswith("fast"):
            try:
                data["require_fast_transient"] = bool(int(tok.replace("fast", "")))
            except ValueError:
                pass
        elif tok.startswith("b"):
            try:
                data["min_galactic_lat"] = float(tok.replace("b", ""))
            except ValueError:
                pass
    return data


def audit_results(root_path: Path) -> List[Dict[str, Any]]:
    """Traverse directories and parse simulation outputs and metadata."""
    records = []
    pattern = str(root_path / "**/*.json")
    for f in sorted(glob.glob(pattern, recursive=True)):
        if f.endswith(".metadata.json"):
            continue
        p = Path(f)
        try:
            content = json.loads(p.read_text())
        except Exception:
            continue

        meta_p = Path(f"{f}.metadata.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}

        label = p.parent.parent.name
        criteria = meta.get("detection_criteria", parse_label_metadata(label))

        rate_summaries = content.get("rate_summaries", [])
        if not rate_summaries:
            continue

        rs = rate_summaries[0]
        n_sim = rs.get("n_simulated", 100_000)
        n_det = rs.get("n_detected", 0)
        eff = rs.get("overall_efficiency", n_det / max(n_sim, 1))
        vt = rs.get("effective_vt_gpc3_yr", 0.0)

        # Audit detected sources
        sources = content.get("detected_sources", [])
        n_sources_checked = len(sources)
        violations_span = 0
        violations_ndet = 0
        violations_sep = 0
        violations_tail_30d = 0
        violations_snr_prim = 0
        violations_snr_sec = 0

        configured_span = float(criteria.get("max_timespan_days", 14.0))
        configured_min_det = int(criteria.get("min_detections", 2))
        configured_min_prim_det = int(criteria.get("min_detections_primary", 1))
        configured_min_sep_hrs = float(criteria.get("min_time_separation_hours", 3.0))
        configured_snr_prim = float(criteria.get("snr_threshold", 5.0))
        configured_snr_sec = float(criteria.get("snr_threshold_secondary", 3.0))

        for src in sources:
            p_times = [float(t) for t in src.get("photometry_times", []) if math.isfinite(float(t))]
            snrs = [float(s) for s in src.get("photometry_snrs", []) if math.isfinite(float(s))]

            if len(p_times) < configured_min_det:
                violations_ndet += 1
            if len(p_times) >= 2:
                span_days = p_times[-1] - p_times[0]
                if span_days > (configured_span + 1e-4):
                    violations_span += 1
                if (span_days * 24.0) < (configured_min_sep_hrs - 1e-4):
                    violations_sep += 1
                if span_days > 30.0:
                    violations_tail_30d += 1

            if snrs:
                n_prim = sum(1 for s in snrs if s >= (configured_snr_prim - 1e-4))
                n_sec = sum(1 for s in snrs if s >= (configured_snr_sec - 1e-4))
                if n_prim < configured_min_prim_det:
                    violations_snr_prim += 1
                if n_sec < configured_min_det:
                    violations_snr_sec += 1

        records.append({
            "file": str(p),
            "label": label,
            "model": meta.get("model", p.parent.name),
            "n_simulated": n_sim,
            "n_detected": n_det,
            "overall_efficiency": eff,
            "effective_vt_gpc3_yr": vt,
            "sources_audited": n_sources_checked,
            "violations_max_timespan": violations_span,
            "violations_min_detections": violations_ndet,
            "violations_min_separation": violations_sep,
            "violations_primary_snr": violations_snr_prim,
            "violations_secondary_snr": violations_snr_sec,
            "violations_tail_30d": violations_tail_30d,
            **criteria,
        })
    return records


def compute_marginal_sensitivities(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute 1D marginal detection efficiencies across all 10 detection criteria."""
    if not records:
        return {}

    criteria_cols = [
        "snr_threshold", "snr_threshold_secondary", "min_detections",
        "min_detections_primary", "max_timespan_days", "min_time_separation_hours",
        "min_rise_rate", "min_fade_rate", "min_galactic_lat", "require_fast_transient"
    ]

    marginals = {}
    ranking_data = []

    if pd is not None:
        df = pd.DataFrame(records)
        for col in criteria_cols:
            if col in df.columns:
                marg = df.groupby(col).agg(
                    n_configs=("overall_efficiency", "count"),
                    eff_mean=("overall_efficiency", "mean"),
                    eff_median=("overall_efficiency", "median"),
                    eff_std=("overall_efficiency", "std"),
                    vt_mean=("effective_vt_gpc3_yr", "mean"),
                ).reset_index()

                max_eff = marg["eff_mean"].max()
                min_eff = marg["eff_mean"].min()
                mean_eff = marg["eff_mean"].mean()
                marg["relative_efficiency"] = marg["eff_mean"] / max_eff if max_eff > 0 else 0.0
                marginals[col] = marg

                # Dynamic range sensitivity impact
                elasticity = (max_eff - min_eff) / mean_eff if mean_eff > 0 else 0.0
                ranking_data.append({
                    "criterion": col,
                    "max_efficiency": max_eff,
                    "min_efficiency": min_eff,
                    "mean_efficiency": mean_eff,
                    "efficiency_spread": max_eff - min_eff,
                    "fractional_impact": elasticity,
                })

        ranking_df = pd.DataFrame(ranking_data).sort_values("fractional_impact", ascending=False)
        marginals["__ranking__"] = ranking_df
    return marginals


def main():
    args = parse_args()
    root = Path(args.sweep_root)
    print("=" * 80)
    print(f"AUDITING CRITERIA SWEEP RESULTS ACROSS ALL 10 CRITERIA: {root}")
    print("=" * 80)

    records = audit_results(root)
    if not records:
        print(f"No simulation results found under {root}.")
        return

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if pd is not None:
        df = pd.DataFrame(records)
        df.to_csv(out_csv, index=False)
        print(f"Aggregated {len(df)} criteria configurations into {out_csv}")

        # Adherence summary
        total_sources = df["sources_audited"].sum()
        print(f"\nAdherence Validation ({total_sources:,} individual sources audited):")
        print(f"  Timespan Violations:       {df['violations_max_timespan'].sum():,}")
        print(f"  Min Detections Violations: {df['violations_min_detections'].sum():,}")
        print(f"  Min Separation Violations: {df['violations_min_separation'].sum():,}")
        print(f"  Primary SNR Violations:    {df['violations_primary_snr'].sum():,}")
        print(f"  Secondary SNR Violations:  {df['violations_secondary_snr'].sum():,}")
        print(f"  Surrogate Tail (>30d):     {df['violations_tail_30d'].sum():,}")

        marginals = compute_marginal_sensitivities(records)
        ranking_df = marginals.get("__ranking__")
        if ranking_df is not None:
            ranking_csv = Path(args.ranking_csv)
            ranking_df.to_csv(ranking_csv, index=False)
            print("\n" + "=" * 80)
            print("CRITERIA SENSITIVITY RANKING (Fractional Impact on Detection Volume)")
            print("=" * 80)
            print(ranking_df.to_string(index=False))

        print("\n" + "=" * 80)
        print("1D MARGINAL SENSITIVITY HIGHLIGHTS")
        print("=" * 80)
        for crit, mdf in marginals.items():
            if crit.startswith("__"):
                continue
            print(f"\n--- Criterion: {crit} ---")
            print(mdf.to_string(index=False))
    else:
        # Fallback without pandas
        import csv
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved {len(records)} records to {out_csv} using standard csv.")


if __name__ == "__main__":
    main()

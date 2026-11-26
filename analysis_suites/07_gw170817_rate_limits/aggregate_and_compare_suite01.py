#!/usr/bin/env python3
"""Aggregate Suite 07 GW170817 rate limits and compare against Suite 01.

Purpose
-------
Parses serialized ``survey_sim`` JSON outputs from Suite 07 (fixed GW170817/AT2017gfo
archetypes), writes ``rate_limit_summary.csv``, and generates pairwise comparison metrics
and ``COMPARISON_REPORT.md`` against Suite 01.

CLI Example
-----------
    python analysis_suites/07_gw170817_rate_limits/aggregate_and_compare_suite01.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ANDREONI_DURATION_YRS = 23.0 / 12.0
ANDREONI_FIXED_95 = 1775.0
ANDREONI_ISOTROPIC_95 = 4029.0

_SUITE07_MODELS = (
    "GW170817_Bu2026_FixedView",
    "GW170817_Bu2026_Isotropic",
    "GW170817_Metzger_Blue",
    "GW170817_Metzger_Mean",
)

_PAIRWISE_SPECS = (
    ("GW170817_Bu2026_FixedView", "Bu2026Fixed", "Numerical Bu2026 at Fixed Polar Viewing Angle (iota=26 deg)"),
    ("GW170817_Bu2026_Isotropic", "Bu2026Vary", "Numerical Bu2026 with Isotropic Viewing Angles (Angle-Averaged)"),
    ("GW170817_Metzger_Blue", "Metzger", "Analytic Metzger Model: GW170817 Blue Peak vs. Suite 01 Model"),
    ("GW170817_Metzger_Mean", "Metzger", "Analytic Metzger Model: GW170817 Mean Ejecta vs. Suite 01 Model"),
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Aggregate Suite 07 GW170817 simulation results and compare against Suite 01.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--suite07-dir", default="results/07_gw170817_rate_limits",
                    help="Directory containing Suite 07 simulation JSON files.")
    ap.add_argument("--suite01-csv", default="results/01_large_scale_rate_limits/rate_limit_summary.csv",
                    help="Primary Suite 01 aggregated CSV.")
    ap.add_argument("--suite01-fallback-csv", default="results/rate_limit_benchmark.csv",
                    help="Fallback CSV if primary Suite 01 summary is absent.")
    ap.add_argument("--output-dir", default="results/07_gw170817_rate_limits",
                    help="Directory to write aggregated tables and comparison report.")
    return ap.parse_args()


def extract_model_from_json(path: Path, content: Any) -> str:
    """Identify Suite 07 model name from ``.metadata.json`` or ``path``."""
    meta_path = Path(f"{path}.metadata.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if "model" in meta:
                return meta["model"]
        except Exception:
            pass
    path_str = str(path)
    for m in _SUITE07_MODELS:
        if m in path_str:
            return m
    return "Unknown"


def process_suite07_file(path: Path) -> List[Dict[str, Any]]:
    """Extract rate summary records from a single Suite 07 JSON file."""
    if path.name.endswith(".metadata.json"):
        return []
    try:
        content = json.loads(path.read_text())
    except Exception:
        return []

    model = extract_model_from_json(path, content)
    rate_summaries = content.get("rate_summaries", []) if isinstance(content, dict) else (content if isinstance(content, list) else [])
    rows: List[Dict[str, Any]] = []
    for rs in rate_summaries:
        eff_vt = float(rs.get("effective_vt_gpc3_yr", float("nan")))
        if not math.isfinite(eff_vt) or eff_vt <= 0:
            continue
        duration = float(rs.get("survey_duration_years", 8.0057))
        r90, r95, r99 = -math.log(0.10) / eff_vt, -math.log(0.05) / eff_vt, -math.log(0.01) / eff_vt
        r95_scaled = r95 * (duration / ANDREONI_DURATION_YRS)
        n_sim = int(rs.get("n_simulated", content.get("n_simulated", 1_000_000)))
        n_det = int(rs.get("n_detected", content.get("n_detected", 0)))
        eff = float(rs.get("overall_efficiency", n_det / max(n_sim, 1)))
        rows.append({
            "file": str(path),
            "suite": "Suite 07 (GW170817 Fixed)",
            "model": model,
            "transient_type": rs.get("transient_type", "Kilonova"),
            "n_simulated": n_sim,
            "n_detected": n_det,
            "overall_efficiency": eff,
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
            "ratio_to_andreoni_fixed_1775": r95_scaled / ANDREONI_FIXED_95,
            "ratio_to_andreoni_isotropic_4029": r95_scaled / ANDREONI_ISOTROPIC_95,
        })
    return rows


def load_suite01_data(primary_path: Path, fallback_path: Path) -> pd.DataFrame:
    """Load Suite 01 rate-limit summary CSV and harmonize column aliases."""
    csv_file = primary_path if primary_path.exists() else fallback_path
    if not csv_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_file)
    if "model" not in df.columns:
        df["model"] = "Unknown"
        if "file" in df.columns:
            df.loc[df["file"].str.contains("Bu2026Fixed|ztf_fixed|bu2026_fixed", case=False, na=False), "model"] = "Bu2026Fixed"
            df.loc[df["file"].str.contains("Bu2026Vary|bu2026_vary", case=False, na=False), "model"] = "Bu2026Vary"
            df.loc[df["file"].str.contains("Metzger", case=False, na=False), "model"] = "Metzger"
    if "rate_95_upper_limit" not in df.columns and "rate_95_zero_events" in df.columns:
        df["rate_95_upper_limit"] = df["rate_95_zero_events"]
    if "rate_95_zero_events" not in df.columns and "rate_95_upper_limit" in df.columns:
        df["rate_95_zero_events"] = df["rate_95_upper_limit"]
    df["suite"] = "Suite 01 (Fiducial/Distribution)"
    return df


def _summarize_models(df: pd.DataFrame, include_scaled: bool = False) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if df.empty:
        return out
    for m, sub in df.groupby("model", sort=False):
        if m in ("Other", "Unknown"):
            continue
        rec: Dict[str, Any] = {
            "n_runs": len(sub),
            "vt_mean": float(sub["effective_vt_gpc3_yr"].mean()),
            "vt_std": float(sub["effective_vt_gpc3_yr"].std()) if len(sub) > 1 else 0.0,
            "r95_mean": float(sub["rate_95_upper_limit"].mean()),
            "r95_std": float(sub["rate_95_upper_limit"].std()) if len(sub) > 1 else 0.0,
            "eff_mean": float(sub["overall_efficiency"].mean()) if "overall_efficiency" in sub.columns else float("nan"),
        }
        if include_scaled and "duration_scaled_95" in sub.columns:
            rec["duration_scaled_95_mean"] = float(sub["duration_scaled_95"].mean())
        out[str(m)] = rec
    return out


def perform_comparison(df_suite07: pd.DataFrame, df_suite01: pd.DataFrame) -> Dict[str, Any]:
    """Compute per-model summaries, Suite 07 vs. Suite 01 ratios, and viewing-angle suppression."""
    s07_sum = _summarize_models(df_suite07, include_scaled=True)
    s01_sum = _summarize_models(df_suite01, include_scaled=False)
    pairwise = []
    for m07, m01, desc in _PAIRWISE_SPECS:
        s07, s01 = s07_sum.get(m07), s01_sum.get(m01)
        if s07 and s01 and s01["vt_mean"] > 0 and s07["r95_mean"] > 0:
            pairwise.append({
                "suite07_model": m07,
                "suite01_model": m01,
                "description": desc,
                "suite07_vt_gpc3_yr": s07["vt_mean"],
                "suite01_vt_gpc3_yr": s01["vt_mean"],
                "vt_ratio_07_to_01": s07["vt_mean"] / s01["vt_mean"],
                "suite07_r95": s07["r95_mean"],
                "suite01_r95": s01["r95_mean"],
                "r95_ratio_07_to_01": s07["r95_mean"] / s01["r95_mean"],
                "r95_percent_diff": ((s07["r95_mean"] - s01["r95_mean"]) / s01["r95_mean"]) * 100.0,
            })

    vas: Dict[str, Any] = {}
    s07_fix, s07_iso = s07_sum.get("GW170817_Bu2026_FixedView"), s07_sum.get("GW170817_Bu2026_Isotropic")
    if s07_fix and s07_iso and s07_iso["vt_mean"] > 0:
        vas = {
            "fixed_vt": s07_fix["vt_mean"],
            "isotropic_vt": s07_iso["vt_mean"],
            "vt_suppression_factor": s07_fix["vt_mean"] / s07_iso["vt_mean"],
            "fixed_r95": s07_fix["r95_mean"],
            "isotropic_r95": s07_iso["r95_mean"],
            "r95_increase_factor": s07_iso["r95_mean"] / s07_fix["r95_mean"],
        }
    return {"suite07_summary": s07_sum, "suite01_summary": s01_sum,
            "pairwise_comparisons": pairwise, "viewing_angle_suppression": vas}


def generate_markdown_report(metrics: Dict[str, Any], output_path: Path) -> None:
    """Write a concise Markdown comparison table of Suite 07 vs. Suite 01."""
    lines = [
        "# Suite 07 (GW170817 Fixed) vs. Suite 01 Comparison",
        "",
        "| Pair | Suite 07 `<VT>` [Gpc^3 yr] | Suite 01 `<VT>` [Gpc^3 yr] | `<VT>` Ratio | Suite 07 R95 | Suite 01 R95 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for p in metrics.get("pairwise_comparisons", []):
        lines.append(
            f"| `{p['suite07_model']}` vs `{p['suite01_model']}` | "
            f"{p['suite07_vt_gpc3_yr']:.4e} | {p['suite01_vt_gpc3_yr']:.4e} | "
            f"{p['vt_ratio_07_to_01']:.2f}x | {p['suite07_r95']:,.1f} | {p['suite01_r95']:,.1f} |"
        )
    vas = metrics.get("viewing_angle_suppression", {})
    if vas:
        lines.extend([
            "",
            "## Viewing-Angle Suppression (GW170817 Ejecta)",
            f"- Polar (`26 deg`) `<VT>`: `{vas['fixed_vt']:.4e}` Gpc^3 yr (`R95 = {vas['fixed_r95']:,.1f}`)",
            f"- Isotropic (`cos iota ~ U[0,1]`) `<VT>`: `{vas['isotropic_vt']:.4e}` Gpc^3 yr (`R95 = {vas['isotropic_r95']:,.1f}`)",
            f"- Polar-over-isotropic `<VT>` ratio: `{vas['vt_suppression_factor']:.2f}x`",
        ])
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    suite07_dir, out_dir = Path(args.suite07_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows07 = [
        r for f in sorted(suite07_dir.glob("**/*.json"))
        if not f.name.endswith(".metadata.json")
        for r in process_suite07_file(f)
    ]
    if not rows07:
        print(f"[Notice] No simulation JSON files found in {suite07_dir}.")
        return

    df07 = pd.DataFrame(rows07)
    df07.to_csv(out_dir / "rate_limit_summary.csv", index=False)
    df01 = load_suite01_data(Path(args.suite01_csv), Path(args.suite01_fallback_csv))
    metrics = perform_comparison(df07, df01)
    (out_dir / "suite07_vs_suite01_comparison.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if metrics["pairwise_comparisons"]:
        pd.DataFrame(metrics["pairwise_comparisons"]).to_csv(out_dir / "suite07_vs_suite01_comparison.csv", index=False)
    generate_markdown_report(metrics, out_dir / "COMPARISON_REPORT.md")


if __name__ == "__main__":
    main()

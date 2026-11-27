"""Generate ``paper/numbers.tex`` LaTeX macros directly from compiled snapshots.

Why this module exists
----------------------
``paper_figures.ipynb`` updates ``paper/table1_master_rate_limits.tex``, whereas inline prose
numbers in ``paper/ms.tex`` (volume-time, baseline upper limits, 23-month vs. 8-year growth,
distance quantiles, viewing-angle ratios, and Monte Carlo convergence errors) previously had
to be edited by hand. Emitting ``paper/numbers.tex`` from the snapshots guarantees prose and
tables never drift out of sync when ensembles are re-run on restricted searched exposures.

Note: TeX control sequences permit ASCII letters only (``A-Za-z``), so every macro name below
is strictly alphabetical.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import analysis as A
from .notebook import Context, all_limits, setup
from .snapshot import REPO_ROOT

DEFAULT_MACROS_PATH = REPO_ROOT / "paper" / "numbers.tex"
DEFAULT_PAPER2_MACROS_PATH = REPO_ROOT / "paper" / "numbers_paper2.tex"
DEFAULT_SUITE09_DIR = REPO_ROOT / "results" / "09_imposter_contamination"
POLAR_MODEL = "Bu2026Fixed"
ISO_MODEL = "Bu2026Vary"


def compute_paper1_macros(ctx: Optional[Context] = None, snapshot: str = "latest") -> Dict[str, str]:
    """Compute all Paper I LaTeX macro strings (strictly alphabetical names) from ``ctx``."""
    if ctx is None:
        ctx = setup(snapshot, quiet=True)

    runs = ctx.runs
    lim = ctx.lim
    polar = lim.loc[POLAR_MODEL]
    iso = lim.loc[ISO_MODEL]

    dur = float(runs.loc[runs["model"] == POLAR_MODEL, "survey_duration_years"].iloc[0])
    cmp = A.andreoni_comparison(ctx.tm, runs, [POLAR_MODEL, ISO_MODEL]).set_index("model")
    dq_polar = A.distance_quantiles(ctx.zh, POLAR_MODEL, q=(0.5, 0.9, 0.99))
    cvt_polar = A.cumulative_vt_vs_redshift(ctx.zh, runs, POLAR_MODEL)
    z99_polar = float(np.interp(0.99, cvt_polar["vt_fraction_below"], cvt_polar["z_hi"]))

    inc = A.inclination_efficiency(ctx.src, runs, ISO_MODEL, n_bins=20).sort_values("iota_deg_lo")
    face_on = inc.iloc[0]
    edge_on = inc.iloc[-1]
    xc = A.inclination_crosscheck(inc, lim, ISO_MODEL, POLAR_MODEL)

    s_iso = ctx.src[ctx.src["model"] == ISO_MODEL]
    frac_rec_gt45 = 100.0 * float((np.degrees(s_iso["inclination_rad"]) > 45.0).mean())
    frac_pop_gt45 = 100.0 * float(np.cos(np.radians(45.0)))

    conv_p = A.subsampling_convergence(runs, POLAR_MODEL, [1_000_000], n_draws=1000).iloc[0]
    conv_i = A.subsampling_convergence(runs, ISO_MODEL, [1_000_000], n_draws=1000).iloc[0]
    mc_pool_p = 100.0 / float(np.sqrt(polar["n_detected"]))
    mc_pool_i = 100.0 / float(np.sqrt(iso["n_detected"]))

    # Unfiltered Suite 07 (-0.1 dex M_dyn robustness check)
    lim_all = all_limits(ctx.snap.path or snapshot)
    s07_p = lim_all.loc["GW170817_Bu2026_FixedView"]
    s07_i = lim_all.loc["GW170817_Bu2026_Isotropic"]
    dvt_p_pct = 100.0 * (float(s07_p["vt_gpc3_yr"]) / float(polar["vt_gpc3_yr"]) - 1.0)
    dvt_i_pct = 100.0 * (float(s07_i["vt_gpc3_yr"]) / float(iso["vt_gpc3_yr"]) - 1.0)

    macros = {
        "SurveyDurationYr": f"{dur:.3f}",
        "VTPolar": f"{float(polar['vt_gpc3_yr']):.4f}",
        "VTPolarMilli": f"{float(polar['vt_gpc3_yr']) * 1e3:.2f}",
        "RlimPolarIdeal": f"{float(polar['r95']):.0f}",
        "VTIsotropic": f"{float(iso['vt_gpc3_yr']):.4f}",
        "VTIsotropicMilli": f"{float(iso['vt_gpc3_yr']) * 1e3:.2f}",
        "RlimIsotropicIdeal": f"{float(iso['r95']):.0f}",
        "PolarOverIsoRatio": f"{float(polar['vt_gpc3_yr']) / float(iso['vt_gpc3_yr']):.1f}",
        "PolarZNinetyNine": f"{z99_polar:.3f}",
        "PolarDLNinetyNineMpc": f"{dq_polar['dL99']:.0f}",
        "RlimPolarWindow": f"{float(cmp.loc[POLAR_MODEL, 'r95_window']):.0f}",
        "RlimIsoWindow": f"{float(cmp.loc[ISO_MODEL, 'r95_window']):.0f}",
        "VTGainFullOverWindow": f"{float(cmp.loc[POLAR_MODEL, 'vt_gain_full_over_window']):.1f}",
        "DurationGainFullOverWindow": f"{float(cmp.loc[POLAR_MODEL, 'duration_gain']):.1f}",
        "AndreoniProjPolar": f"{round(float(cmp.loc[POLAR_MODEL, 'published_projected_full']), -1):.0f}",
        "AndreoniProjIso": f"{round(float(cmp.loc[ISO_MODEL, 'published_projected_full']), -1):.0f}",
        "VTFaceOn": f"{float(face_on['vt']):.3f}",
        "RlimFaceOn": f"{float(face_on['r95']):.0f}",
        "VTEdgeOnSci": f"{float(edge_on['vt']):.0e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}",
        "IsoPopGtFortyFivePct": f"{frac_pop_gt45:.0f}",
        "IsoRecGtFortyFivePct": f"{frac_rec_gt45:.0f}",
        "IsoInterpDiffPct": f"{abs(100.0 * xc['rel_diff']):.2f}",
        "SuiteSevenDeltaVTPolarPct": f"{dvt_p_pct:+.1f}",
        "SuiteSevenDeltaVTIsoPct": f"{dvt_i_pct:+.0f}",
        "SuiteSevenRlimPolarIdeal": f"{float(s07_p['r95']):.0f}",
        "SuiteSevenRlimIsoIdeal": f"{float(s07_i['r95']):.0f}",
        "MCScatterOneMPolarPct": f"{100.0 * float(conv_p['rel_scatter_vt']):.1f}",
        "MCScatterOneMIsoPct": f"{100.0 * float(conv_i['rel_scatter_vt']):.1f}",
        "MCPooledErrPolarPct": f"{mc_pool_p:.2f}",
        "MCPooledErrIsoPct": f"{mc_pool_i:.2f}",
    }
    for k in macros:
        if not k.isalpha():
            raise ValueError(f"LaTeX macro name {k!r} must contain ASCII letters only")
    return macros


def write_paper1_macros(
    out_path: Path = DEFAULT_MACROS_PATH,
    ctx: Optional[Context] = None,
    snapshot: str = "latest",
) -> Path:
    """Write ``paper/numbers.tex`` from snapshot quantities."""
    macros = compute_paper1_macros(ctx=ctx, snapshot=snapshot)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "%% Auto-generated by `python -m knrates.macros` from snapshots/knrates_results_latest.zip.",
        "%% Do not edit by hand; re-run `python -m knrates.macros` or `notebooks/paper_figures.ipynb`.",
    ]
    for name, val in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def compute_paper2_macros(
    ctx: Optional[Context] = None,
    snapshot: str = "latest",
    suite09_dir: Path = DEFAULT_SUITE09_DIR,
) -> Dict[str, str]:
    """Compute Companion Paper II LaTeX macros (strictly alphabetical names) from Suite 02 + Suite 09."""
    import pandas as pd

    if ctx is None:
        ctx = setup(snapshot, quiet=True)

    gs = ctx.snap["suite02_grid_size"].set_index("model")
    fid = ctx.snap["suite02_fiducial"].set_index("model")
    s1 = ctx.snap["suite02_slices_1d"]
    nc = ctx.snap["suite02_named_configs"]
    resp = A.criteria_response(s1, fid)
    rp = resp[resp["model"] == POLAR_MODEL].set_index("criterion")
    ri = resp[resp["model"] == ISO_MODEL].set_index("criterion")

    vt_s02_p = float(fid.loc[POLAR_MODEL, "effective_vt_gpc3_yr"])
    vt_s02_i = float(fid.loc[ISO_MODEL, "effective_vt_gpc3_yr"])
    vt_s01_p = float(ctx.lim.loc[POLAR_MODEL, "vt_gpc3_yr"])
    vt_s01_i = float(ctx.lim.loc[ISO_MODEL, "vt_gpc3_yr"])

    nc_piv_r95 = nc.pivot(index="regime", columns="model", values="rate_upper_95")
    nc_piv_vt = nc.pivot(index="regime", columns="model", values="effective_vt_gpc3_yr")
    n3_loss_p = 100.0 * (1.0 - float(nc_piv_vt.loc["N_min = 3", POLAR_MODEL]) / vt_s02_p)
    n3_loss_i = 100.0 * (1.0 - float(nc_piv_vt.loc["N_min = 3", ISO_MODEL]) / vt_s02_i)

    macros: Dict[str, str] = {
        "GridPointsTotal": f"{int(gs.loc[POLAR_MODEL, 'n_grid_points']):,}".replace(",", "\\,"),
        "SuiteTwoSamplesPerModel": f"{int(gs.loc[POLAR_MODEL, 'n_samples_max'])}",
        "SuiteTwoFidVTPolarMilli": f"{vt_s02_p * 1e3:.2f}",
        "SuiteTwoFidVTIsoMilli": f"{vt_s02_i * 1e3:.2f}",
        "SuiteTwoOverOnePolarPct": f"{100.0 * vt_s02_p / vt_s01_p:.1f}",
        "SuiteTwoOverOneIsoPct": f"{100.0 * vt_s02_i / vt_s01_i:.1f}",
        "FadeRangePolar": f"{float(rp.loc['min_fade_rate', 'vt_max_over_min']):.1f}",
        "FadeRangeIso": f"{float(ri.loc['min_fade_rate', 'vt_max_over_min']):.1f}",
        "SecSNRRangePolar": f"{float(rp.loc['snr_threshold_secondary', 'vt_max_over_min']):.2f}",
        "SecSNRRangeIso": f"{float(ri.loc['snr_threshold_secondary', 'vt_max_over_min']):.2f}",
        "SpanRangePolar": f"{float(rp.loc['max_timespan_days', 'vt_max_over_min']):.2f}",
        "NminRangePolar": f"{float(rp.loc['min_detections', 'vt_max_over_min']):.2f}",
        "PrimSNRRangePolar": f"{float(rp.loc['snr_threshold', 'vt_max_over_min']):.2f}",
        "ElastSecSNRPolar": f"{float(rp.loc['snr_threshold_secondary', 'elasticity']):.2f}",
        "ElastSecSNRIso": f"{float(ri.loc['snr_threshold_secondary', 'elasticity']):.2f}",
        "ElastPrimSNRPolar": f"{float(rp.loc['snr_threshold', 'elasticity']):.2f}",
        "ElastPrimSNRIso": f"{float(ri.loc['snr_threshold', 'elasticity']):.2f}",
        "ElastFadePolar": f"{float(rp.loc['min_fade_rate', 'elasticity']):.2f}",
        "ElastFadeIso": f"{float(ri.loc['min_fade_rate', 'elasticity']):.2f}",
        "NamedRlimMinPolar": f"{float(nc_piv_r95[POLAR_MODEL].min()):.0f}",
        "NamedRlimMaxPolar": f"{float(nc_piv_r95[POLAR_MODEL].max()):.0f}",
        "NamedRlimSpanPolar": f"{float(nc_piv_r95[POLAR_MODEL].max() / nc_piv_r95[POLAR_MODEL].min()):.1f}",
        "NamedRlimMinIso": f"{float(nc_piv_r95[ISO_MODEL].min()):.0f}",
        "NamedRlimMaxIso": f"{float(nc_piv_r95[ISO_MODEL].max()):.0f}",
        "NminThreeVTLossPolarPct": f"{n3_loss_p:.1f}",
        "NminThreeVTLossIsoPct": f"{n3_loss_i:.1f}",
    }

    suite09_dir = Path(suite09_dir)
    p_9a = suite09_dir / "wls_slope_and_rise_systematic.csv"
    p_fr = suite09_dir / "suite09_contamination_frontier.csv"
    p_co = suite09_dir / "suite09_color_summary.csv"
    if p_9a.exists():
        df9a = pd.read_csv(p_9a)
        pf_p = df9a[(df9a["model"] == POLAR_MODEL) & (df9a["regime"] == "Fiducial_PureFade")].iloc[0]
        r0_p = df9a[(df9a["model"] == POLAR_MODEL) & (df9a["regime"] == "Fiducial_Rise0_Suite01")].iloc[0]
        nf_p = df9a[(df9a["model"] == POLAR_MODEL) & (df9a["regime"] == "No_Fade")].iloc[0]
        pf_i = df9a[(df9a["model"] == ISO_MODEL) & (df9a["regime"] == "Fiducial_PureFade")].iloc[0]
        r0_i = df9a[(df9a["model"] == ISO_MODEL) & (df9a["regime"] == "Fiducial_Rise0_Suite01")].iloc[0]
        rise0_pct_p = 100.0 * (float(r0_p["engine_direct_noiseless_n"]) / max(1.0, float(pf_p["engine_direct_noiseless_n"])) - 1.0)
        rise0_pct_i = 100.0 * (float(r0_i["engine_direct_noiseless_n"]) / max(1.0, float(pf_i["engine_direct_noiseless_n"])) - 1.0)
        nofade_diff_p = 100.0 * abs(float(nf_p["noisy_over_noiseless_ratio"]) - 1.0)
        macros.update({
            "NoFadeNoisyRatioPolar": f"{float(nf_p['noisy_over_noiseless_ratio']):.3f}",
            "NoFadeFilterMinusEnginePct": f"{nofade_diff_p:.1f}",
            "FidNoisyMinusCleanPolarPct": f"{float(pf_p['noisy_minus_noiseless_pct']):+.1f}",
            "FidNoisyMinusCleanIsoPct": f"{float(pf_i['noisy_minus_noiseless_pct']):+.1f}",
            "RiseZeroLeakPolarPct": f"{rise0_pct_p:+.1f}",
            "RiseZeroLeakIsoPct": f"{rise0_pct_i:+.1f}",
        })
    if p_fr.exists():
        fr = pd.read_csv(p_fr)
        sub = fr[(fr["signal_model"] == ISO_MODEL) & (fr["model_variant"] == "hybrid_fiesta")]
        r_loose = sub[(sub["regime"] == "Loose_NoFade") & (sub["color_gate"] == "none")].iloc[0]
        r_fid = sub[(sub["regime"] == "Fiducial") & (sub["color_gate"] == "none")].iloc[0]
        r_fid_red = sub[(sub["regime"] == "Fiducial") & (sub["color_gate"] == "require_reddening_015")].iloc[0]
        r_strict_red = sub[(sub["regime"] == "Strict_Fade05") & (sub["color_gate"] == "require_reddening_025")].iloc[0]
        red_drop_pct = 100.0 * (1.0 - float(r_fid_red["n_imposters_expected"]) / float(r_fid["n_imposters_expected"]))
        strict_drop_pct = 100.0 * (1.0 - float(r_strict_red["n_imposters_expected"]) / float(r_fid["n_imposters_expected"]))
        macros.update({
            "ImpYieldLooseEightYr": f"{float(r_loose['n_imposters_expected']):,.0f}".replace(",", "\\,"),
            "ImpYieldFidEightYr": f"{float(r_fid['n_imposters_expected']):,.0f}".replace(",", "\\,"),
            "ImpYieldFidLoEightYr": f"{float(r_fid['n_imposters_lo']):,.0f}".replace(",", "\\,"),
            "ImpYieldFidHiEightYr": f"{float(r_fid['n_imposters_hi']):,.0f}".replace(",", "\\,"),
            "ImpYieldSNIIbFid": f"{float(r_fid.get('n_SNIIb', 0.0)):,.0f}".replace(",", "\\,"),
            "ImpYieldFBOTFid": f"{float(r_fid.get('n_FBOT', 0.0)):,.0f}".replace(",", "\\,"),
            "ImpDominantFid": str(r_fid["dominant_imposter"]),
            "PurityFidNoColorPct": f"{100.0 * float(r_fid['purity']):.3f}",
            "PurityFidRedGatePct": f"{100.0 * float(r_fid_red['purity']):.3f}",
            "CompFidRedGatePct": f"{100.0 * float(r_fid_red['completeness_vs_fiducial']):.0f}",
            "ImpYieldFidRedEightYr": f"{float(r_fid_red['n_imposters_expected']):.0f}",
            "ImpRedGateReductionPct": f"{red_drop_pct:.0f}",
            "ImpYieldStrictRedEightYr": f"{float(r_strict_red['n_imposters_expected']):.0f}",
            "ImpStrictRedReductionPct": f"{strict_drop_pct:.0f}",
            "PurityStrictRedPct": f"{100.0 * float(r_strict_red['purity']):.3f}",
            "ParetoPointCount": f"{int(sub['pareto_optimal'].sum())}",
        })
    if p_co.exists():
        co = pd.read_csv(p_co)
        kn_i = co[co["transient_class"] == ISO_MODEL].iloc[0]
        kn_p = co[co["transient_class"] == POLAR_MODEL].iloc[0]
        macros.update({
            "KNColorSlopePolar": f"{float(kn_p['median_dgr_dt']):+.2f}",
            "KNColorSlopeIso": f"{float(kn_i['median_dgr_dt']):+.2f}",
        })

    for k in macros:
        if not k.isalpha():
            raise ValueError(f"LaTeX macro name {k!r} must contain ASCII letters only")
    return macros


def write_paper2_macros(
    out_path: Path = DEFAULT_PAPER2_MACROS_PATH,
    ctx: Optional[Context] = None,
    snapshot: str = "latest",
    suite09_dir: Path = DEFAULT_SUITE09_DIR,
) -> Path:
    """Write ``paper/numbers_paper2.tex`` from Suite 02 snapshot + Suite 09 exploratory campaign."""
    macros = compute_paper2_macros(ctx=ctx, snapshot=snapshot, suite09_dir=suite09_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "%% Auto-generated by `python -m knrates.macros --paper2`.",
        "%% Do not edit by hand; re-run `python -m knrates.macros --all` or `notebooks/companion/paper2_figures.ipynb`.",
    ]
    for name, val in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", default="latest")
    ap.add_argument("--out", default=str(DEFAULT_MACROS_PATH))
    ap.add_argument("--paper2", action="store_true", help="Write paper/numbers_paper2.tex instead of numbers.tex")
    ap.add_argument("--all", action="store_true", help="Write both numbers.tex and numbers_paper2.tex")
    args = ap.parse_args(argv)
    if args.all:
        p1 = write_paper1_macros(Path(args.out), snapshot=args.snapshot)
        p2 = write_paper2_macros(DEFAULT_PAPER2_MACROS_PATH, snapshot=args.snapshot)
        print(f"[knrates.macros] wrote {p1} and {p2}")
        return 0
    if args.paper2:
        p2 = write_paper2_macros(DEFAULT_PAPER2_MACROS_PATH if args.out == str(DEFAULT_MACROS_PATH) else Path(args.out), snapshot=args.snapshot)
        print(f"[knrates.macros] wrote {p2}")
        return 0
    p = write_paper1_macros(Path(args.out), snapshot=args.snapshot)
    print(f"[knrates.macros] wrote {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

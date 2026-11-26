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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", default="latest")
    ap.add_argument("--out", default=str(DEFAULT_MACROS_PATH))
    args = ap.parse_args(argv)
    p = write_paper1_macros(Path(args.out), snapshot=args.snapshot)
    print(f"[knrates.macros] wrote {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

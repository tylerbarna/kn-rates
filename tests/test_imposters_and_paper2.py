"""Unit tests for Companion Paper II imposter contamination, color metrics, Pareto frontier, and literature rates."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from knrates import imposters as IMP
from knrates import literature as L


def test_imposter_rates_table_all_verified_with_ordered_intervals():
    df = L.imposter_rates_table()
    assert len(df) >= 8
    assert set(df["transient_class"]) >= {
        "SNIa", "SNII", "SNIIb", "SNIax", "CSMInteraction", "ShockedCocoon", "FBOT", "TDE"
    }
    for r in df.itertuples(index=False):
        assert r.verified is True
        assert 0.0 < r.low <= r.rate <= r.high
        assert len(str(r.arxiv)) >= 9


def test_lightcurve_color_metrics_distinguishes_reddening_kilonova_from_blue_fbot():
    # Kilonova: at t=0, g=19.0, r=19.0 (g-r = 0.0); at t=2.0 d, g=21.2, r=20.4 (g-r = +0.8) -> +0.40 mag/d!
    kn = IMP.lightcurve_color_metrics(
        times=[60000.0, 60000.05, 60002.0, 60002.05],
        mags=[19.0, 19.0, 21.2, 20.4],
        errs=[0.05, 0.05, 0.10, 0.08],
        bands=["g", "r", "g", "r"],
    )
    assert kn["n_gr_pairs"] == 2
    assert kn["initial_g_minus_r"] == pytest.approx(0.0, abs=1e-6)
    assert kn["late_g_minus_r"] == pytest.approx(0.8, abs=1e-6)
    assert kn["color_slope_gr_per_day"] == pytest.approx(0.4, rel=1e-2)

    # FBOT: stays blue (g - r = -0.25 at both epochs) -> slope ~ 0.0 mag/d
    fbot = IMP.lightcurve_color_metrics(
        times=[60000.0, 60000.02, 60002.0, 60002.02],
        mags=[18.5, 18.75, 20.0, 20.25],
        errs=[0.05, 0.05, 0.08, 0.08],
        bands=["g", "r", "g", "r"],
    )
    assert fbot["n_gr_pairs"] == 2
    assert fbot["initial_g_minus_r"] == pytest.approx(-0.25, abs=1e-6)
    assert fbot["color_slope_gr_per_day"] == pytest.approx(0.0, abs=1e-6)

    # Single-band or >0.5 d unpaired detections must return NaN slope cleanly
    unpaired = IMP.lightcurve_color_metrics(
        times=[60000.0, 60003.0],
        mags=[19.0, 20.5],
        errs=[0.05, 0.08],
        bands=["g", "r"],
    )
    assert unpaired["n_gr_pairs"] == 0
    assert math.isnan(unpaired["color_slope_gr_per_day"])


def test_pareto_optimal_mask_identifies_exact_frontier():
    # A=(1.0, 0.2), B=(0.8, 0.6), C=(0.5, 0.9) are Pareto-optimal;
    # D=(0.7, 0.5) is dominated by B; E=(0.8, 0.2) is dominated by A and B.
    comp = np.array([1.0, 0.8, 0.5, 0.7, 0.8, np.nan])
    pur = np.array([0.2, 0.6, 0.9, 0.5, 0.2, 0.99])
    mask = IMP.pareto_optimal_mask(comp, pur)
    assert mask.tolist() == [True, True, True, False, False, False]


def test_evaluate_contamination_table_propagates_rates_and_purity_bounds():
    vt = pd.DataFrame([
        {"regime": "Fiducial", "transient_class": "Bu2026Vary", "model_variant": "fiesta", "color_gate": "none", "vt_gpc3_yr": 0.004},
        {"regime": "Fiducial", "transient_class": "SNIIb", "model_variant": "fiesta", "color_gate": "none", "vt_gpc3_yr": 0.001},
        {"regime": "Fiducial", "transient_class": "FBOT", "model_variant": "fiesta", "color_gate": "none", "vt_gpc3_yr": 0.005},
        {"regime": "Strict", "transient_class": "Bu2026Vary", "model_variant": "fiesta", "color_gate": "red", "vt_gpc3_yr": 0.003},
        {"regime": "Strict", "transient_class": "SNIIb", "model_variant": "fiesta", "color_gate": "red", "vt_gpc3_yr": 0.00005},
        {"regime": "Strict", "transient_class": "FBOT", "model_variant": "fiesta", "color_gate": "red", "vt_gpc3_yr": 0.0001},
    ])
    out = IMP.evaluate_contamination_table(vt, signal_model="Bu2026Vary", signal_rate_gpc3_yr=100.0)
    assert len(out) == 2

    fid = out.iloc[0]
    # Signal = 100 * 0.004 = 0.4; SNIIb = 7000 * 0.001 = 7.0; FBOT = 40 * 0.005 = 0.2 -> total imp = 7.2
    assert fid["n_signal_expected"] == pytest.approx(0.4)
    assert fid["n_imposters_expected"] == pytest.approx(7.2)
    assert fid["dominant_imposter"] == "SNIIb"
    assert fid["purity"] == pytest.approx(0.4 / 7.6)
    assert fid["purity_lo"] < fid["purity"] < fid["purity_hi"]
    assert bool(fid["pareto_optimal"]) is True
    assert bool(out.iloc[1]["pareto_optimal"]) is True


def test_paper2_macros_and_suite09_figures_render(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from knrates import figures as F
    from knrates import macros as M

    out_tex = tmp_path / "numbers_paper2.tex"
    p = M.write_paper2_macros(out_path=out_tex)
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "\\newcommand{\\GridPointsTotal}{427\\,692}" in content
    assert "\\newcommand{\\SuiteTwoSamplesPerModel}{100}" in content

    s09_dir = M.DEFAULT_SUITE09_DIR
    if (s09_dir / "wls_slope_and_rise_systematic.csv").exists():
        df9a = pd.read_csv(s09_dir / "wls_slope_and_rise_systematic.csv")
        fr = pd.read_csv(s09_dir / "suite09_contamination_frontier.csv")
        co = pd.read_csv(s09_dir / "suite09_color_summary.csv")
        fig7 = F.wls_slope_systematic(df9a)
        fig8 = F.imposter_frontier_and_colors(fr, co)
        plt.close(fig7)
        plt.close(fig8)

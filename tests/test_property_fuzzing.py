"""Randomized property-based fuzzing tests for ``knrates`` mathematical invariants.

Tests exact algebraic, symmetry, and monotonicity properties across randomized inputs:
  1. ``band_wls_rates``: time-shift invariance, band-label permutation invariance, and exact
     recovery of noiseless linear fade/rise slopes;
  2. ``evaluate`` / ``passes``: strict monotonicity under post-qualification threshold tightening;
  3. ``lightcurve_color_metrics``: time-translation invariance, common-zeropoint shift invariance,
     and exact linear ``d(g - r)/dt`` slope recovery;
  4. ``pareto_optimal_mask``: strict dominance soundness and completeness on 2D random clouds;
  5. ``clopper_pearson`` and ``summarize_runs``: interval containment and Jensen's inequality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from knrates import criteria as KC
from knrates import imposters as IMP
from knrates import stats as KS


def test_band_wls_rates_shift_symmetry_and_linear_recovery():
    rng = np.random.default_rng(20261126)
    for _ in range(120):
        t0 = float(rng.uniform(58000.0, 61000.0))
        dt_shift = float(rng.uniform(-500.0, 500.0))
        alpha_fade = float(rng.uniform(0.05, 1.5))
        t_grid = t0 + np.cumsum(rng.uniform(0.25, 1.5, size=5))
        mags = 18.2 + alpha_fade * (t_grid - t_grid[0])
        snrs = rng.uniform(4.0, 25.0, size=5)
        pts = [(float(t), float(s), float(m), "g") for t, s, m in zip(t_grid, snrs, mags)]
        pts_shifted = [(float(t + dt_shift), float(s), float(m), "r") for t, s, m in zip(t_grid, snrs, mags)]

        fade1, rise1 = KC.band_wls_rates(pts)
        fade2, rise2 = KC.band_wls_rates(pts_shifted)
        assert abs(fade1 - alpha_fade) < 1e-6
        assert rise1 == 0.0
        assert abs(fade1 - fade2) < 1e-10
        assert abs(rise1 - rise2) < 1e-10


def test_criteria_post_qualifying_monotonicity():
    rng = np.random.default_rng(8888)
    base = {**KC.SUITE02_FIDUCIAL, "min_rise_rate": 99.0}
    tighter = [
        {**base, "snr_threshold": 5.8},
        {**base, "min_detections": 3},
        {**base, "min_detections_primary": 2},
        {**base, "max_timespan_days": 7.0},
        {**base, "min_time_separation_hours": 6.0},
        {**base, "min_fade_rate": 0.5},
    ]
    bands = np.array(["g", "r", "i"])
    for _ in range(250):
        n = int(rng.integers(1, 10))
        ts = np.cumsum(rng.uniform(0.05, 3.0, size=n))
        snrs = rng.uniform(2.2, 15.0, size=n)
        mags = 18.5 + rng.uniform(-0.5, 2.0, size=n)
        bs = rng.choice(bands, size=n)
        pts = [(float(t), float(s), float(m), str(b)) for t, s, m, b in zip(ts, snrs, mags, bs)]
        base_ok = KC.passes(pts, base)
        for cfg_t in tighter:
            if KC.passes(pts, cfg_t):
                assert base_ok


def test_lightcurve_color_metrics_invariances_and_exact_slope():
    rng = np.random.default_rng(5150)
    for _ in range(100):
        n_epochs = int(rng.integers(3, 7))
        t_epochs = 60000.0 + np.cumsum(rng.uniform(0.6, 1.8, size=n_epochs))
        c0 = float(rng.uniform(-0.3, 0.4))
        beta = float(rng.uniform(-0.2, 0.6))
        r_mags = 18.4 + 0.35 * (t_epochs - t_epochs[0])
        g_mags = r_mags + c0 + beta * (t_epochs - t_epochs[0])

        times = np.concatenate([t_epochs, t_epochs + 0.02])
        mags = np.concatenate([g_mags, r_mags])
        errs = np.full(2 * n_epochs, 0.08)
        bands = ["g"] * n_epochs + ["r"] * n_epochs

        cm1 = IMP.lightcurve_color_metrics(times, mags, errs, bands)
        # Shift all times by +123.4 d and all magnitudes in both bands by +0.77 mag
        cm2 = IMP.lightcurve_color_metrics(times + 123.4, mags + 0.77, errs, bands)

        assert cm1["n_gr_pairs"] == n_epochs
        assert abs(cm1["color_slope_gr_per_day"] - beta) < 0.01
        assert abs(cm1["initial_g_minus_r"] - cm2["initial_g_minus_r"]) < 1e-10
        assert abs(cm1["late_g_minus_r"] - cm2["late_g_minus_r"]) < 1e-10
        assert abs(cm1["color_slope_gr_per_day"] - cm2["color_slope_gr_per_day"]) < 1e-10


def test_pareto_optimal_mask_soundness_and_completeness():
    rng = np.random.default_rng(2026)
    for _ in range(80):
        n = int(rng.integers(5, 60))
        comp = rng.uniform(0.0, 1.5, size=n)
        pur = rng.uniform(0.0, 1.0, size=n)
        mask = IMP.pareto_optimal_mask(comp, pur)
        assert mask.any()
        opt_idx = np.where(mask)[0]
        sub_idx = np.where(~mask)[0]

        # Soundness: no Pareto-optimal point is strictly dominated by another point
        for i in opt_idx:
            dom = (comp >= comp[i] - 1e-9) & (pur >= pur[i] - 1e-9) & ((comp > comp[i] + 1e-9) | (pur > pur[i] + 1e-9))
            dom[i] = False
            assert not np.any(dom)

        # Completeness: every rejected finite point is strictly dominated by at least one Pareto-optimal point
        for j in sub_idx:
            dom_by_opt = (
                (comp[opt_idx] >= comp[j] - 1e-9)
                & (pur[opt_idx] >= pur[j] - 1e-9)
                & ((comp[opt_idx] > comp[j] + 1e-9) | (pur[opt_idx] > pur[j] + 1e-9))
            )
            assert np.any(dom_by_opt)


def test_clopper_pearson_and_jensen_inequality():
    rng = np.random.default_rng(12345)
    for _ in range(60):
        n = int(rng.integers(10, 500))
        k = int(rng.integers(0, n + 1))
        lo, hi = KS.clopper_pearson(k, n, cl=0.90)
        eff = k / n
        assert 0.0 <= lo <= eff <= hi <= 1.0

    # Symmetric positive realizations => mean(1/VT) >= 1/mean(VT) (Jensen's inequality for convex 1/x)
    vts = np.array([0.010, 0.012, 0.014, 0.016, 0.018])
    df = pd.DataFrame({
        "model": ["M"] * 5,
        "vt_gpc3_yr": vts,
        "n_simulated": [100_000] * 5,
        "n_detected": [100, 120, 140, 160, 180],
    })
    s = KS.summarize_runs(df).iloc[0]
    mean_of_r95 = float(np.mean(KS.LN20 / vts))
    assert mean_of_r95 > float(s["r95"])

"""Analytical unit tests for knrates.stats, knrates.analysis, and knrates.literature."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from knrates import analysis as A
from knrates import literature as L
from knrates import stats as S


def _poisson_cdf(n_obs: int, mu: float) -> float:
    return math.exp(-mu) * sum((mu ** k) / math.factorial(k) for k in range(n_obs + 1))


def test_poisson_factor_and_rate_upper_limit_exact_identities() -> None:
    with pytest.raises(ValueError, match="cl"):
        S.poisson_factor(0.0)
    with pytest.raises(ValueError, match="cl"):
        S.poisson_factor(1.0)

    for cl in (0.90, 0.95, 0.99):
        for n_obs in (0, 1, 3, 7):
            mu = S.poisson_factor(cl=cl, n_obs=n_obs)
            assert _poisson_cdf(n_obs, mu) == pytest.approx(1.0 - cl, abs=1e-11)

    arr = S.rate_upper_limit([0.0, -1.0, 2.0], cl=0.95, n_obs=0)
    assert np.isnan(arr[0]) and np.isnan(arr[1])
    assert arr[2] == pytest.approx(-math.log(0.05) / 2.0, rel=1e-12)


def test_clopper_pearson_binned_efficiency_and_hypergeom_subsampling() -> None:
    lo0, hi0 = S.clopper_pearson(0, 10, cl=0.6827)
    assert lo0 == 0.0
    assert (1.0 - hi0) ** 10 == pytest.approx((1.0 - 0.6827) / 2.0, abs=1e-11)

    lon, hin = S.clopper_pearson(10, 10, cl=0.6827)
    assert hin == 1.0
    assert lon ** 10 == pytest.approx((1.0 - 0.6827) / 2.0, abs=1e-11)

    lo2, hi2 = S.clopper_pearson(2, 5, cl=0.90)
    tail_lo = sum(math.comb(5, j) * (lo2 ** j) * ((1.0 - lo2) ** (5 - j)) for j in range(2, 6))
    tail_hi = sum(math.comb(5, j) * (hi2 ** j) * ((1.0 - hi2) ** (5 - j)) for j in range(0, 3))
    assert tail_lo == pytest.approx(0.05, abs=1e-11)
    assert tail_hi == pytest.approx(0.05, abs=1e-11)

    df_eff = S.binned_efficiency([0.1, 0.2, 0.8], np.array([0.0, 0.5, 1.0]), np.array([10, 5]))
    assert list(df_eff["n_detected"]) == [2, 1]
    assert list(df_eff["efficiency"]) == pytest.approx([0.2, 0.2])

    with pytest.raises(ValueError, match="n_sub"):
        S.hypergeom_subsample_counts(3, 10, 11, size=5)
    draws = S.hypergeom_subsample_counts(40, 100, 25, size=5000, rng=np.random.default_rng(42))
    assert draws.min() >= 0 and draws.max() <= 25
    assert draws.mean() == pytest.approx(10.0, abs=0.15)


def test_pooled_vt_weights_by_n_simulated_and_matches_running_limit() -> None:
    runs = pd.DataFrame({
        "model": ["M1", "M1", "M2"],
        "task_id": ["000", "001", "000"],
        "n_simulated": [1000, 3000, 2000],
        "n_detected": [10, 90, 20],
        "vt_gpc3_yr": [0.01, 0.03, 0.02],
        "z_max": [0.2, 0.2, 0.2],
        "survey_duration_years": [7.5, 7.5, 7.5],
        "survey_omega_sr": [7.0, 7.0, 7.0],
    })
    p = S.pooled_vt(runs[runs["model"] == "M1"])
    # Injection-weighted mean: (1000*0.01 + 3000*0.03) / 4000 = 0.025 (unweighted would be 0.020)
    assert p["vt"] == pytest.approx(0.025, rel=1e-12)
    assert p["efficiency"] == pytest.approx(100 / 4000, rel=1e-12)

    hl = A.headline_limits(runs, models=["M2", "M1"])
    assert list(hl["model"]) == ["M2", "M1"]
    m1_hl = hl[hl["model"] == "M1"].iloc[0]
    assert m1_hl["vt_gpc3_yr"] == pytest.approx(0.025, rel=1e-12)

    rpl = A.running_pooled_limit(runs, "M1")
    assert rpl.iloc[-1]["vt"] == pytest.approx(m1_hl["vt_gpc3_yr"], rel=1e-12)
    assert rpl.iloc[-1]["r95"] == pytest.approx(m1_hl["r95"], rel=1e-12)


def test_cosmology_distances_and_inclination_bin_conservation() -> None:
    assert A.comoving_distance_mpc(0.0)[0] == pytest.approx(0.0, abs=1e-12)
    # Hubble law at z = 1e-4: d_C -> c * z / H0
    z_small = 1e-4
    dc_small = A.comoving_distance_mpc(z_small)[0]
    assert dc_small == pytest.approx(299792.458 * z_small / A.H0_SURVEY_SIM, rel=2e-4)

    zs = np.array([0.05, 0.10, 0.25])
    dc = A.comoving_distance_mpc(zs)
    dl = A.luminosity_distance_mpc(zs)
    vc = A.comoving_volume_gpc3(zs)
    np.testing.assert_allclose(dl, (1.0 + zs) * dc, rtol=1e-12)
    assert np.all(np.diff(vc) > 0)

    # Conservation identity: integral_0^1 VT(cos iota) d(cos iota) == VT_iso
    rng = np.random.default_rng(7)
    cos_det = rng.beta(2.5, 1.0, size=200)  # polar-enhanced detections in [0, 1]
    sources = pd.DataFrame({
        "model": ["Iso"] * 200,
        "task_id": ["000"] * 100 + ["001"] * 100,
        "inclination_rad": np.arccos(cos_det),
    })
    runs = pd.DataFrame({
        "model": ["Iso", "Iso", "Fixed"],
        "task_id": ["000", "001", "000"],
        "n_simulated": [5000, 5000, 5000],
        "n_detected": [100, 100, 145],
        "vt_gpc3_yr": [0.020, 0.020, 0.029],
        "z_max": [0.2, 0.2, 0.2],
        "survey_duration_years": [7.5, 7.5, 7.5],
        "survey_omega_sr": [7.0, 7.0, 7.0],
    })
    inc = A.inclination_efficiency(sources, runs, "Iso", n_bins=5)
    lim = A.headline_limits(runs).set_index("model")
    chk = A.inclination_crosscheck(inc, lim, "Iso", "Fixed", iota_rad=0.45)
    assert chk["vt_iso_from_bins"] == pytest.approx(chk["vt_iso_pooled"], rel=1e-11)

    # Monthly VT partition sums exactly to total VT
    tm = pd.DataFrame({
        "model": ["Iso", "Iso", "Iso"],
        "month": ["2019-01", "2019-06", "2021-03"],
        "n_detected": [50, 70, 80],
    })
    by_ep = A.vt_by_epoch(tm, runs, "Iso")
    assert by_ep["vt_cumulative"].iloc[-1] == pytest.approx(0.020, rel=1e-12)
    win = A.vt_in_window(tm, runs, "Iso", "2019-01", "2019-12")
    assert win["vt"] == pytest.approx(0.020 * (120 / 200), rel=1e-12)


def test_literature_lookup_aliases_and_validation() -> None:
    r_new = L.lookup("andreoni2020_gw170817")
    r_old = L.lookup("andreoni20_gw170817")
    assert r_new == r_old
    assert r_new["limit"] == 1775.0

    sel = L.select(["gwtc4", "gwtc3_bns_rate"], kind="gw_rates")
    assert [x["key"] for x in sel] == ["gwtc4_bns_rate", "gwtc3_bns_rate"]
    assert "Gpc^-3 yr^-1" in L.describe("andreoni2020_gw170817")
    assert "BNS merger rate" in L.describe("gwtc4_bns_rate")

    with pytest.raises(KeyError):
        L.lookup("nonexistent_paper_2099")
    with pytest.raises(KeyError):
        L.select(["andreoni2020_gw170817"], kind="gw_rates")

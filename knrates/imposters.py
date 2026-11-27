"""Companion Paper II (Suite 09) imposter contamination, color evolution, and Pareto-frontier utilities.

Purpose & Scientific Role
-------------------------
Provides pure, dependency-light analyses for Companion Paper II
(``paper/companion_paper_imposters_criteria.tex``):

1. :func:`lightcurve_color_metrics` extracts quasi-simultaneous ``g - r`` color at peak, late-time
   ``g - r`` color, and ``d(g - r)/dt`` reddening rate [mag/day] from multi-band photometry.
   Lanthanide-rich kilonovae cool rapidly (``d(g - r)/dt >= +0.20..+0.50 mag/d``), whereas shock
   cooling, FBOTs, and CSM imposters evolve much more slowly in color.
2. :func:`evaluate_contamination_table` combines per-class simulated ``<VT>(C)`` across selection
   regimes with primary-source volumetric rates (:data:`knrates.literature.IMPOSTER_RATES`) to
   compute expected 8-year ZTF contaminant yields, uncertainty intervals, and photometric purity.
3. :func:`pareto_optimal_mask` identifies non-dominated selection configurations on the 2D
   ``(completeness, purity)`` plane.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import literature as L
from .criteria import snr_from_mag_err


def lightcurve_color_metrics(
    times: Sequence[float],
    mags: Sequence[float],
    errs: Sequence[float],
    bands: Sequence[str],
    *,
    snr_min: float = 3.0,
    max_pair_dt_days: float = 0.5,
    min_baseline_days: float = 0.4,
) -> Dict[str, float]:
    """Compute quasi-simultaneous ``g - r`` colors and ``d(g - r)/dt`` [mag/d] for one light curve."""
    g_pts: List[Tuple[float, float, float]] = []
    r_pts: List[Tuple[float, float, float]] = []
    for t, m, e, b in zip(times, mags, errs, bands):
        tf, mf, ef = float(t), float(m), float(e)
        if not (math.isfinite(tf) and math.isfinite(mf) and math.isfinite(ef) and ef > 0):
            continue
        snr = snr_from_mag_err(ef)
        if snr < snr_min - 1e-4:
            continue
        bs = str(b)[-1]
        if bs == "g":
            g_pts.append((tf, mf, ef))
        elif bs == "r":
            r_pts.append((tf, mf, ef))

    if not g_pts or not r_pts:
        return {
            "n_gr_pairs": 0,
            "initial_g_minus_r": float("nan"),
            "late_g_minus_r": float("nan"),
            "color_slope_gr_per_day": float("nan"),
        }

    g_pts.sort(key=lambda x: x[0])
    r_pts.sort(key=lambda x: x[0])

    # Pair each g detection with its nearest r detection within max_pair_dt_days
    pairs: List[Tuple[float, float, float]] = []
    used_r: set[int] = set()
    for tg, mg, eg in g_pts:
        best_j: Optional[int] = None
        best_dt = max_pair_dt_days + 1.0
        for j, (tr, _mr, _er) in enumerate(r_pts):
            if j in used_r:
                continue
            dt = abs(tg - tr)
            if dt <= max_pair_dt_days and dt < best_dt:
                best_dt = dt
                best_j = j
        if best_j is not None:
            used_r.add(best_j)
            tr, mr, er = r_pts[best_j]
            pairs.append((0.5 * (tg + tr), mg - mr, math.hypot(eg, er)))

    if not pairs:
        return {
            "n_gr_pairs": 0,
            "initial_g_minus_r": float("nan"),
            "late_g_minus_r": float("nan"),
            "color_slope_gr_per_day": float("nan"),
        }

    pairs.sort(key=lambda x: x[0])
    init_gr = float(pairs[0][1])
    late_gr = float(pairs[-1][1])
    slope = float("nan")
    if len(pairs) >= 2 and (pairs[-1][0] - pairs[0][0]) >= min_baseline_days:
        t_arr = np.array([p[0] for p in pairs], dtype=float)
        c_arr = np.array([p[1] for p in pairs], dtype=float)
        w_arr = np.array([1.0 / max(p[2] ** 2, 1e-4) for p in pairs], dtype=float)
        w_sum = float(np.sum(w_arr))
        t_bar = float(np.sum(w_arr * t_arr) / w_sum)
        c_bar = float(np.sum(w_arr * c_arr) / w_sum)
        denom = float(np.sum(w_arr * (t_arr - t_bar) ** 2))
        if denom > 1e-12:
            slope = float(np.sum(w_arr * (t_arr - t_bar) * (c_arr - c_bar)) / denom)

    return {
        "n_gr_pairs": len(pairs),
        "initial_g_minus_r": init_gr,
        "late_g_minus_r": late_gr,
        "color_slope_gr_per_day": slope,
    }


def evaluate_contamination_table(
    vt_table: pd.DataFrame,
    *,
    signal_model: str = "Bu2026Vary",
    signal_rate_gpc3_yr: float = 100.0,
    imposter_rates: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Combine per-regime ``vt_gpc3_yr`` across signal + imposter classes into expected yields and purity.

    Parameters
    ----------
    vt_table : DataFrame
        Columns required: ``["regime", "transient_class", "model_variant", "vt_gpc3_yr"]``
        (plus optional ``"color_gate"``).
    signal_model : str
        ``transient_class`` name of the kilonova signal row per ``(regime, model_variant, color_gate)``.
    signal_rate_gpc3_yr : float
        Reference astrophysical BNS merger rate [Gpc^-3 yr^-1] (default: 100 Gpc^-3 yr^-1, near the
        geometric center of the LVK GWTC-4.0 90% interval 7.6-250 Gpc^-3 yr^-1).
    """
    rates_df = imposter_rates if imposter_rates is not None else L.imposter_rates_table()
    rate_map = rates_df.set_index("transient_class")[["rate", "low", "high", "label"]].to_dict("index")

    df = vt_table.copy()
    if "color_gate" not in df.columns:
        df["color_gate"] = "none"

    sig_rows = df[df["transient_class"] == signal_model].copy()
    fid_mask = (sig_rows["regime"] == "Fiducial") & (sig_rows["color_gate"] == "none")
    vt_fid = float(sig_rows.loc[fid_mask, "vt_gpc3_yr"].iloc[0]) if fid_mask.any() else float(sig_rows["vt_gpc3_yr"].max())

    imp_rows = df[df["transient_class"] != signal_model].copy()
    out_records: List[Dict[str, Any]] = []

    for (regime, variant, cgate), s_grp in sig_rows.groupby(["regime", "model_variant", "color_gate"], sort=False):
        vt_sig = float(s_grp["vt_gpc3_yr"].iloc[0])
        n_sig = signal_rate_gpc3_yr * vt_sig
        sub_imp = imp_rows[
            (imp_rows["regime"] == regime)
            & (imp_rows["model_variant"] == variant)
            & (imp_rows["color_gate"] == cgate)
        ]

        n_imp_mid = n_imp_lo = n_imp_hi = 0.0
        dominant_cls = "none"
        dominant_n = -1.0
        class_yields: Dict[str, float] = {}

        for r in sub_imp.itertuples(index=False):
            tcls = str(r.transient_class)
            if tcls not in rate_map:
                continue
            rm = rate_map[tcls]
            vt_i = float(r.vt_gpc3_yr)
            ym = float(rm["rate"]) * vt_i
            yl = float(rm["low"]) * vt_i
            yh = float(rm["high"]) * vt_i
            n_imp_mid += ym
            n_imp_lo += yl
            n_imp_hi += yh
            class_yields[f"n_{tcls}"] = ym
            if ym > dominant_n:
                dominant_n = ym
                dominant_cls = tcls

        denom = n_sig + n_imp_mid
        purity = (n_sig / denom) if denom > 0 else float("nan")
        purity_lo = (n_sig / (n_sig + n_imp_hi)) if (n_sig + n_imp_hi) > 0 else float("nan")
        purity_hi = (n_sig / (n_sig + n_imp_lo)) if (n_sig + n_imp_lo) > 0 else float("nan")

        out_records.append({
            "regime": regime,
            "model_variant": variant,
            "color_gate": cgate,
            "signal_model": signal_model,
            "signal_rate_gpc3_yr": signal_rate_gpc3_yr,
            "vt_signal_gpc3_yr": vt_sig,
            "completeness_vs_fiducial": (vt_sig / vt_fid) if vt_fid > 0 else float("nan"),
            "n_signal_expected": n_sig,
            "n_imposters_expected": n_imp_mid,
            "n_imposters_lo": n_imp_lo,
            "n_imposters_hi": n_imp_hi,
            "imposter_to_signal_ratio": (n_imp_mid / n_sig) if n_sig > 0 else float("inf"),
            "purity": purity,
            "purity_lo": purity_lo,
            "purity_hi": purity_hi,
            "dominant_imposter": dominant_cls,
            **class_yields,
        })

    out = pd.DataFrame(out_records)
    if not out.empty:
        out["pareto_optimal"] = False
        for (_sm, _mv), idx in out.groupby(["signal_model", "model_variant"], sort=False).groups.items():
            sub = out.loc[idx]
            out.loc[idx, "pareto_optimal"] = pareto_optimal_mask(
                sub["completeness_vs_fiducial"].to_numpy(float),
                sub["purity"].fillna(0.0).to_numpy(float),
            )
    return out


def pareto_optimal_mask(completeness: np.ndarray, purity: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """True for points not strictly dominated in ``(completeness, purity)`` (both maximized)."""
    c = np.asarray(completeness, dtype=float)
    p = np.asarray(purity, dtype=float)
    n = len(c)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not (math.isfinite(c[i]) and math.isfinite(p[i])):
            keep[i] = False
            continue
        dominated = ((c >= c[i] - tol) & (p >= p[i] - tol)) & ((c > c[i] + tol) | (p > p[i] + tol))
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
    return keep

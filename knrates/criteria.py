"""Canonical kilonova photometric selection criteria shared across kn-rates and real-ztf-search.

Both the Suite 02 simulate-once/filter-many sweep in ``kn-rates`` and Stage C of the real archival
search in ``real-ztf-search`` (``knsearch.criteria``) import ``band_wls_rates`` and ``evaluate`` from
this module so simulation efficiency and real-data selection are guaranteed to use identical math.

A photometric point is a tuple ``(t_mjd, snr, mag, band)`` where ``snr = 1.08574 / sigma_mag`` (or
``survey_sim``'s depth S/N ``5 * 10**(0.4 * (diffmaglim - mag))``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# Numerical tolerance used across grid and single-object threshold comparisons.
EPS = 1e-4

Point = Tuple[float, float, float, str]

#: Canonical 9-key fiducial criteria of the Suite 01 / Suite 07 production runs.
FIDUCIAL_CRITERIA: Dict[str, Any] = dict(
    snr_threshold=5.0,
    snr_threshold_secondary=3.0,
    min_detections=2,
    min_detections_primary=1,
    max_timespan_days=14.0,
    min_time_separation_hours=3.0,
    min_rise_rate=0.0,
    min_fade_rate=0.3,
    min_galactic_lat=15.0,
)

#: Full 10-key Suite 02 fiducial point (adds explicit ``require_fast_transient`` flag).
SUITE02_FIDUCIAL: Dict[str, Any] = {
    **FIDUCIAL_CRITERIA,
    "require_fast_transient": True,
}


def band_wls_rates(pts, dedup_days: float = 2.0 / 24.0, min_seg_span: float = 0.01):
    """Per-band rise/fade rates mirroring survey_sim's fast-transient test.

    For each band: flux-average points closer than ``dedup_days``, split at the brightest
    point and fit weighted least-squares slopes (sigma = 1.0857/SNR) to the pre-peak (rise)
    and post-peak (fade) segments (>= 2 points spanning > ``min_seg_span`` d).
    Returns (max fade slope [mag/d, >0 = fading], max rise rate [mag/d, >0 = brightening]).
    """
    by_band: Dict[str, List[Tuple[float, float, float]]] = {}
    for p in pts:
        by_band.setdefault(p[3], []).append((p[0], p[1], p[2]))
    best_fade, best_rise = 0.0, 0.0
    for rows in by_band.values():
        rows.sort()
        binned = []  # (t, mag, snr)
        k = 0
        while k < len(rows):
            grp = [rows[k]]
            while k + 1 < len(rows) and rows[k + 1][0] - grp[0][0] <= dedup_days:
                k += 1
                grp.append(rows[k])
            fl = np.array([10 ** (-0.4 * g[2]) for g in grp])
            sn = np.array([g[1] for g in grp])
            t = float(np.mean([g[0] for g in grp]))
            binned.append((t, float(-2.5 * np.log10(fl.mean())), float(np.sqrt(np.sum(sn ** 2)))))
            k += 1
        if len(binned) < 2:
            continue
        ipk = int(np.argmin([b[1] for b in binned]))
        for seg, is_fade in ((binned[ipk:], True), (binned[:ipk + 1], False)):
            if len(seg) < 2 or seg[-1][0] - seg[0][0] <= min_seg_span:
                continue
            t = np.array([s_[0] for s_ in seg]); m = np.array([s_[1] for s_ in seg])
            w = (np.array([s_[2] for s_ in seg]) / 1.0857) ** 2
            tw = np.sum(w * t) / np.sum(w); mw = np.sum(w * m) / np.sum(w)
            den = np.sum(w * (t - tw) ** 2)
            if den <= 0:
                continue
            slope = float(np.sum(w * (t - tw) * (m - mw)) / den)
            if is_fade:
                best_fade = max(best_fade, slope)
            else:
                best_rise = max(best_rise, -slope)
    return best_fade, best_rise


def qualifying(points: Iterable[Point], cfg: Dict[str, Any]) -> List[Point]:
    """Points at or above the secondary S/N threshold, sorted by time."""
    th = cfg["snr_threshold_secondary"] - EPS
    return sorted((p for p in points if p[1] >= th), key=lambda p: p[0])


def evaluate(points: Sequence[Point], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate every criterion separately and return intermediate quantities and flags."""
    q = qualifying(points, cfg)
    n_q = len(q)
    n_prim = sum(1 for p in q if p[1] >= cfg["snr_threshold"] - EPS)
    span = (q[-1][0] - q[0][0]) if n_q >= 2 else 0.0
    if n_q >= 2:
        mfade, mrise = band_wls_rates(q)
    else:
        mfade, mrise = 0.0, 0.0

    min_fade, min_rise = cfg.get("min_fade_rate", 0.0), cfg.get("min_rise_rate", 0.0)
    if cfg.get("require_fast_transient", True) and (min_fade > 0.0 or min_rise > 0.0):
        fast = (min_fade > 0.0 and mfade >= min_fade - EPS) or (min_rise > 0.0 and mrise >= min_rise - EPS)
    else:
        fast = True

    res = {
        "n_qualifying": n_q,
        "n_primary": n_prim,
        "span_days": span,
        "max_fade_rate": mfade,
        "max_rise_rate": mrise,
        "pass_n_detections": n_q >= cfg["min_detections"],
        "pass_n_primary": n_prim >= cfg["min_detections_primary"],
        "pass_span": span <= cfg["max_timespan_days"] + EPS,
        "pass_separation": span * 24.0 >= cfg["min_time_separation_hours"] - EPS,
        "pass_fast": fast,
    }
    if n_q == 1:
        res["pass_span"] = res["pass_separation"] = res["pass_fast"] = cfg["min_detections"] == 1
    res["passes"] = all(res[k] for k in ("pass_n_detections", "pass_n_primary", "pass_span", "pass_separation",
                                          "pass_fast"))
    return res


def passes(points: Sequence[Point], cfg: Dict[str, Any]) -> bool:
    """True if the light curve satisfies the kilonova criteria in ``cfg`` (galactic latitude excluded)."""
    return bool(evaluate(points, cfg)["passes"])


def snr_from_mag_err(sigma_mag: float) -> float:
    """S/N corresponding to a magnitude uncertainty, with the constant used by kn-rates (1.08574)."""
    return 1.08574 / sigma_mag if sigma_mag > 0 else 0.0

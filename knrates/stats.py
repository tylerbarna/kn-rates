"""Statistical helpers for injection-recovery rate limits.

Conventions
-----------
* ``n_det`` / ``n_sim`` are integer counts of recovered / injected transients.
* ``VT`` is the effective (efficiency-weighted) volume-time in Gpc^3 yr.
* With zero real candidates, the classical Poisson upper limit at confidence
  level CL is ``R_CL = -ln(1 - CL) / VT`` (e.g. 2.996/VT at 95%).

Pooling independent Monte Carlo realisations
--------------------------------------------
Every realisation draws ``n_sim`` transients from the same injection volume, so
``VT_i = (n_det_i / n_sim_i) * K`` with a common normalisation ``K`` (volume x
duration).  The minimum-variance estimate of VT from M realisations is therefore
the injection-weighted mean ``VT = sum(VT_i * n_sim_i) / sum(n_sim_i)`` and the
rate limit is ``R = -ln(1-CL) / VT``.  The *median of per-run limits* is a
biased estimator (Jensen's inequality) and should not be used for headline
numbers; it is kept only for diagnostics.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

LN20 = -math.log(0.05)  # 2.995732...


def poisson_factor(cl: float = 0.95, n_obs: int = 0) -> float:
    """Classical (frequentist) Poisson upper-limit factor for ``n_obs`` observed events.

    For ``n_obs == 0`` this is ``-ln(1 - cl)``. For ``n_obs > 0`` (no background)
    it is the upper end of the central/one-sided Neyman interval, computed via
    the chi-square relation ``mu_up = 0.5 * chi2.ppf(cl, 2 * (n_obs + 1))``.
    """
    if not 0.0 < cl < 1.0:
        raise ValueError("cl must be in (0, 1)")
    if n_obs == 0:
        return -math.log(1.0 - cl)
    from scipy.stats import chi2

    return 0.5 * float(chi2.ppf(cl, 2 * (n_obs + 1)))


def rate_upper_limit(vt_gpc3_yr, cl: float = 0.95, n_obs: int = 0):
    """Upper limit on the volumetric rate [Gpc^-3 yr^-1] for effective VT [Gpc^3 yr]."""
    vt = np.asarray(vt_gpc3_yr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(vt > 0, poisson_factor(cl, n_obs) / vt, np.nan)
    return float(out) if out.ndim == 0 else out


def pooled_vt(df: pd.DataFrame, vt_col: str = "vt_gpc3_yr", nsim_col: str = "n_simulated",
              ndet_col: str = "n_detected") -> Dict[str, float]:
    """Injection-weighted pooled VT and its Monte Carlo uncertainty.

    Returns a dict with pooled VT, two uncertainty estimates (binomial from the
    pooled detection count and empirical run-to-run scatter / sqrt(M)), the
    total counts and the number of realisations.
    """
    sub = df[[vt_col, nsim_col, ndet_col]].dropna()
    if sub.empty:
        return dict(vt=np.nan, vt_err_binomial=np.nan, vt_err_empirical=np.nan,
                    n_det=0, n_sim=0, n_runs=0, efficiency=np.nan)
    w = sub[nsim_col].to_numpy(float)
    vt_i = sub[vt_col].to_numpy(float)
    n_det = int(sub[ndet_col].sum())
    n_sim = int(sub[nsim_col].sum())
    vt = float(np.sum(vt_i * w) / np.sum(w))
    eff = n_det / n_sim if n_sim else np.nan
    # Binomial relative error on the pooled efficiency (and hence VT)
    rel_binom = math.sqrt((1.0 - eff) / n_det) if n_det > 0 else np.inf
    m = len(sub)
    vt_err_emp = float(np.std(vt_i, ddof=1) / math.sqrt(m)) if m > 1 else np.nan
    return dict(vt=vt, vt_err_binomial=vt * rel_binom, vt_err_empirical=vt_err_emp,
                n_det=n_det, n_sim=n_sim, n_runs=m, efficiency=eff)


def summarize_runs(runs: pd.DataFrame, by: Sequence[str] = ("model",), cl: float = 0.95) -> pd.DataFrame:
    """Per-group pooled VT, rate limits at 90/95/99% CL, and MC uncertainties."""
    rows = []
    for key, g in runs.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        p = pooled_vt(g)
        r = dict(zip(by, key))
        r.update(
            n_runs=p["n_runs"], n_simulated=p["n_sim"], n_detected=p["n_det"],
            efficiency=p["efficiency"], vt_gpc3_yr=p["vt"],
            vt_err_binomial=p["vt_err_binomial"], vt_err_empirical=p["vt_err_empirical"],
            r90=rate_upper_limit(p["vt"], 0.90), r95=rate_upper_limit(p["vt"], 0.95),
            r99=rate_upper_limit(p["vt"], 0.99),
        )
        # Propagated (1-sigma) MC uncertainty on R95 from the empirical VT scatter
        err = p["vt_err_empirical"] if np.isfinite(p["vt_err_empirical"]) else p["vt_err_binomial"]
        r["r95_err_mc"] = r["r95"] * err / p["vt"] if p["vt"] > 0 else np.nan
        r["median_of_run_r95"] = float(np.nanmedian(rate_upper_limit(g["vt_gpc3_yr"].to_numpy(float), cl)))
        rows.append(r)
    return pd.DataFrame(rows)


def hypergeom_subsample_counts(k_total: int, n_total: int, n_sub: int, size: int,
                               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Detection counts obtained by drawing ``n_sub`` injections (without replacement)
    from a realised simulation with ``k_total`` detections among ``n_total`` injections.

    This is exactly the distribution of the number of detections in a random
    subset of the simulated injections, so it is a *real* sub-sampling of an
    existing simulation (not a synthetic re-draw from an assumed efficiency).
    """
    rng = np.random.default_rng() if rng is None else rng
    if n_sub > n_total:
        raise ValueError("n_sub must be <= n_total")
    return rng.hypergeometric(k_total, n_total - k_total, n_sub, size=size)


def clopper_pearson(k: int, n: int, cl: float = 0.6827) -> tuple:
    """Exact binomial confidence interval for an efficiency k/n."""
    from scipy.stats import beta

    a = 1.0 - cl
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def binned_efficiency(values: Iterable[float], edges: np.ndarray, n_injected_per_bin: np.ndarray,
                      cl: float = 0.6827) -> pd.DataFrame:
    """Efficiency per bin with Clopper-Pearson errors given detected values and injected counts."""
    counts, _ = np.histogram(np.asarray(list(values), dtype=float), bins=edges)
    rows = []
    for i, (k, n) in enumerate(zip(counts, n_injected_per_bin)):
        n = int(round(n))
        lo, hi = clopper_pearson(int(k), n, cl) if n > 0 else (np.nan, np.nan)
        rows.append(dict(lo_edge=edges[i], hi_edge=edges[i + 1], center=0.5 * (edges[i] + edges[i + 1]),
                         n_detected=int(k), n_injected=n, efficiency=k / n if n else np.nan,
                         eff_lo=lo, eff_hi=hi))
    return pd.DataFrame(rows)

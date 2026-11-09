"""Derived analyses shared by the suite notebooks and the paper notebook.

All functions take snapshot tables (see :mod:`knrates.collect`) and return
DataFrames; nothing here contains hard-coded results.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .stats import LN20, binned_efficiency, hypergeom_subsample_counts, rate_upper_limit, summarize_runs

MJD_UNIX_EPOCH = 40587.0


def headline_limits(runs: pd.DataFrame, models: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Pooled VT and R90/R95/R99 per model (injection-weighted over all realisations)."""
    sub = runs if models is None else runs[runs["model"].isin(models)]
    out = summarize_runs(sub, by=("model",))
    first = sub.groupby("model").agg(z_max=("z_max", "first"), duration_yr=("survey_duration_years", "first"),
                                     omega_sr=("survey_omega_sr", "first"))
    out = out.merge(first, left_on="model", right_index=True, how="left")
    if models is not None:
        out["model"] = pd.Categorical(out["model"], categories=list(models), ordered=True)
        out = out.sort_values("model").reset_index(drop=True)
        out["model"] = out["model"].astype(str)
    return out


def running_pooled_limit(runs: pd.DataFrame, model: str, order: str = "task_id") -> pd.DataFrame:
    """R95 from the first k realisations (k = 1..M), to show Monte Carlo convergence."""
    g = runs[runs["model"] == model].copy()
    g["_o"] = pd.to_numeric(g[order], errors="coerce") if order in g else np.arange(len(g))
    g = g.sort_values("_o")
    w = g["n_simulated"].to_numpy(float)
    vt = g["vt_gpc3_yr"].to_numpy(float)
    cum_vt = np.cumsum(vt * w) / np.cumsum(w)
    return pd.DataFrame({"k": np.arange(1, len(g) + 1), "n_injected": np.cumsum(w).astype(np.int64),
                         "n_detected": np.cumsum(g["n_detected"].to_numpy()), "vt": cum_vt,
                         "r95": LN20 / cum_vt})


def vt_by_epoch(texp_month: pd.DataFrame, runs: pd.DataFrame, model: str) -> pd.DataFrame:
    """Cumulative <VT> and R95 vs. survey time, from detections binned by explosion month.

    For injections uniform in explosion time, the VT accumulated over a window W is
    VT_total * N_det(t_exp in W) / N_det(total).  This uses all realisations.
    """
    tm = texp_month[texp_month["model"] == model].copy()
    if tm.empty:
        return pd.DataFrame()
    tm["month"] = pd.PeriodIndex(tm["month"], freq="M")
    tm = tm.groupby("month", as_index=False)["n_detected"].sum().sort_values("month")
    vt_total = summarize_runs(runs[runs["model"] == model])["vt_gpc3_yr"].iloc[0]
    n_tot = tm["n_detected"].sum()
    tm["frac"] = tm["n_detected"] / n_tot
    tm["vt_month"] = vt_total * tm["frac"]
    tm["vt_cumulative"] = tm["vt_month"].cumsum()
    tm["r95_cumulative"] = LN20 / tm["vt_cumulative"]
    tm["month_end"] = tm["month"].dt.to_timestamp(how="end")
    tm["model"] = model
    return tm


def vt_in_window(texp_month: pd.DataFrame, runs: pd.DataFrame, model: str, start: str, end: str) -> dict:
    """<VT> and R95 restricted to explosion epochs in calendar months [start, end] (inclusive, 'YYYY-MM')."""
    tm = vt_by_epoch(texp_month, runs, model)
    sel = (tm["month"] >= pd.Period(start, "M")) & (tm["month"] <= pd.Period(end, "M"))
    vt = float(tm.loc[sel, "vt_month"].sum())
    n_months = int(sel.sum())
    return dict(model=model, start=start, end=end, n_months=n_months,
                n_detected=int(tm.loc[sel, "n_detected"].sum()), vt=vt,
                r95=LN20 / vt if vt > 0 else np.nan)


def inclination_efficiency(sources: pd.DataFrame, runs: pd.DataFrame, model: str,
                           n_bins: int = 10, cos_range=(0.0, 1.0)) -> pd.DataFrame:
    """VT and R95 as a function of viewing angle from an isotropic-inclination ensemble.

    Assumes the injected inclinations are uniform in cos(iota) over ``cos_range``
    (survey_sim's ``vary_inclination=True``).  Each cos-bin then received
    N_inj * dcos / (cos_hi - cos_lo) injections, so the efficiency in the bin is
    exact and VT(iota) = VT_total * (n_bin / n_total) * (cos range / dcos) is the
    VT a population *fixed* at that viewing angle would have.
    """
    s = sources[(sources["model"] == model) & sources["inclination_rad"].notna()]
    r = runs[runs["model"] == model]
    # only use runs whose sources are all stored
    tasks = set(s["task_id"].astype(str))
    r = r[r["task_id"].astype(str).isin(tasks)]
    n_inj = r["n_simulated"].sum()
    width = cos_range[1] - cos_range[0]
    edges = np.linspace(cos_range[0], cos_range[1], n_bins + 1)
    inj = n_inj * np.diff(edges) / width
    eff = binned_efficiency(np.cos(s["inclination_rad"].to_numpy()), edges, inj)
    vt_total = summarize_runs(r)["vt_gpc3_yr"].iloc[0]
    eff_total = r["n_detected"].sum() / n_inj
    k = vt_total / eff_total  # volume-time normalisation (Gpc^3 yr)
    eff["vt"] = eff["efficiency"] * k
    eff["vt_lo"] = eff["eff_lo"] * k
    eff["vt_hi"] = eff["eff_hi"] * k
    eff["r95"] = LN20 / eff["vt"]
    eff["iota_deg_center"] = np.degrees(np.arccos(eff["center"]))
    eff["iota_deg_lo"] = np.degrees(np.arccos(eff["hi_edge"]))
    eff["iota_deg_hi"] = np.degrees(np.arccos(eff["lo_edge"]))
    eff["model"] = model
    eff["n_runs"] = len(r)
    return eff


def subsampling_convergence(runs: pd.DataFrame, model: str, sizes: Iterable[int], n_draws: int = 2000,
                            seed: int = 20261109) -> pd.DataFrame:
    """Monte Carlo convergence from *real* realisations.

    For N <= n_sim of one realisation, the detection count of a random N-subset is
    hypergeometric given that realisation's (k, n).  For N > n_sim we sum
    ceil(N / n_sim) randomly chosen distinct realisations (N must be a multiple).
    """
    rng = np.random.default_rng(seed)
    g = runs[runs["model"] == model]
    k_all = g["n_detected"].to_numpy(int)
    n_all = g["n_simulated"].to_numpy(int)
    vt_all = g["vt_gpc3_yr"].to_numpy(float)
    kfac = float(np.sum(vt_all * n_all) / np.sum(k_all))  # VT per unit efficiency (Gpc^3 yr)
    eff_true = k_all.sum() / n_all.sum()
    rows = []
    for N in sizes:
        n_one = int(n_all[0])
        if N <= n_one:
            idx = rng.integers(0, len(g), size=n_draws)
            ks = np.array([hypergeom_subsample_counts(k_all[i], n_all[i], N, 1, rng)[0] for i in idx])
        else:
            m = int(math.ceil(N / n_one))
            if m > len(g):
                continue
            nd = min(n_draws, 500)
            ks = np.array([k_all[rng.choice(len(g), size=m, replace=False)].sum() for _ in range(nd)])
            N = m * n_one
        eff = ks / N
        vt = eff * kfac
        with np.errstate(divide="ignore"):
            r95 = np.where(vt > 0, LN20 / vt, np.inf)
        finite = np.isfinite(r95)
        rows.append(dict(model=model, n_injected=int(N), n_draws=len(ks), mean_k=float(ks.mean()),
                         p_zero_empirical=float(np.mean(ks == 0)),
                         p_zero_poisson=float(np.exp(-eff_true * N)),
                         rel_scatter_vt=float(np.std(vt, ddof=1) / np.mean(vt)) if np.mean(vt) > 0 else np.nan,
                         rel_err_binomial=float(1 / math.sqrt(eff_true * N)),
                         r95_median=float(np.median(r95[finite])) if finite.any() else np.nan,
                         r95_p16=float(np.percentile(r95[finite], 16)) if finite.any() else np.nan,
                         r95_p84=float(np.percentile(r95[finite], 84)) if finite.any() else np.nan,
                         r95_true=float(LN20 / (eff_true * kfac))))
    return pd.DataFrame(rows)


def comoving_distance_mpc(z, h0: float = 67.66, om: float = 0.30966) -> np.ndarray:
    """Flat LCDM comoving distance (default Planck18 parameters), numerical integration."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.empty_like(z)
    for i, zi in enumerate(z):
        zz = np.linspace(0.0, zi, 513)
        e = np.sqrt(om * (1 + zz) ** 3 + (1 - om))
        out[i] = 299792.458 / h0 * np.trapz(1.0 / e, zz)
    return out


def luminosity_distance_mpc(z, **kw) -> np.ndarray:
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return (1 + z) * comoving_distance_mpc(z, **kw)

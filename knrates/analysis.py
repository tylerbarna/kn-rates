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


# Cosmology used by survey_sim (flat LCDM, h = 0.674, Omega_m = 0.315; survey-sim src/types.rs)
H0_SURVEY_SIM = 67.4
OM_SURVEY_SIM = 0.315


def comoving_distance_mpc(z, h0: float = H0_SURVEY_SIM, om: float = OM_SURVEY_SIM) -> np.ndarray:
    """Flat LCDM comoving distance (default: survey_sim's cosmology), numerical integration."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.empty_like(z)
    for i, zi in enumerate(z):
        zz = np.linspace(0.0, zi, 513)
        e = np.sqrt(om * (1 + zz) ** 3 + (1 - om))
        out[i] = 299792.458 / h0 * np.trapezoid(1.0 / e, zz)
    return out


def luminosity_distance_mpc(z, **kw) -> np.ndarray:
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return (1 + z) * comoving_distance_mpc(z, **kw)


def comoving_volume_gpc3(z, **kw) -> np.ndarray:
    """Full-sky comoving volume [Gpc^3] inside redshift z."""
    return 4.0 / 3.0 * np.pi * (comoving_distance_mpc(z, **kw) / 1e3) ** 3


def efficiency_vs_redshift(z_hist: pd.DataFrame, runs: pd.DataFrame, model: str) -> pd.DataFrame:
    """Recovery efficiency per redshift bin.

    survey_sim injects redshifts with probability proportional to dV_c/dz up to z_max (full sky), so the expected
    number of injections in a bin is N_inj * [V(z_hi) - V(z_lo)] / V(z_max).
    """
    h = z_hist[z_hist["model"] == model].groupby(["z_lo", "z_hi"], as_index=False)["n_detected"].sum()
    r = runs[runs["model"] == model]
    zmax = float(r["z_max"].iloc[0])
    n_inj = float(r["n_simulated"].sum())
    h = h[h["z_hi"] <= zmax + 1e-9].copy()
    v_lo = comoving_volume_gpc3(h["z_lo"].to_numpy())
    v_hi = comoving_volume_gpc3(h["z_hi"].to_numpy())
    vmax = float(comoving_volume_gpc3(zmax)[0])
    h["n_injected"] = n_inj * (v_hi - v_lo) / vmax
    h["efficiency"] = h["n_detected"] / h["n_injected"]
    h["eff_err"] = np.sqrt(h["n_detected"]) / h["n_injected"]
    h["z_center"] = 0.5 * (h["z_lo"] + h["z_hi"])
    h["model"] = model
    return h


def cumulative_vt_vs_redshift(z_hist: pd.DataFrame, runs: pd.DataFrame, model: str) -> pd.DataFrame:
    """Fraction of <VT> contributed by recoveries below each redshift (weights 1/(1+z), as in survey_sim)."""
    h = z_hist[z_hist["model"] == model].groupby(["z_lo", "z_hi"], as_index=False)["n_detected"].sum()
    zc = 0.5 * (h["z_lo"] + h["z_hi"])
    w = h["n_detected"] / (1 + zc)
    h["vt_fraction_below"] = np.cumsum(w) / w.sum()
    h["model"] = model
    return h


# ----------------------------------------------------------------------------------------------------------------
# Quantities shared by the streamlined notebooks (notebooks/0N_*) and the paper notebook.

#: Detection criteria of the Suite 01/07 runs (the Suite 02 fiducial point).
FIDUCIAL_CRITERIA = dict(snr_threshold=5.0, snr_threshold_secondary=3.0, min_detections=2, min_detections_primary=1,
                         max_timespan_days=14.0, min_time_separation_hours=3.0, min_rise_rate=0.0,
                         min_fade_rate=0.3, min_galactic_lat=15.0)

#: Andreoni et al. (2020) used "the first 23 months" of ZTF without exact dates; we adopt these calendar months.
ANDREONI_WINDOW = ("2018-03", "2020-01")

#: Published limit each of our models is compared with (literature keys, see knrates.literature).
ANDREONI_COUNTERPART = {"Bu2026Fixed": "andreoni2020_gw170817", "Bu2026Vary": "andreoni2020_possis_isotropic",
                        "GW170817_Bu2026_FixedView": "andreoni2020_gw170817",
                        "GW170817_Bu2026_Isotropic": "andreoni2020_possis_isotropic"}


def andreoni_comparison(tm: pd.DataFrame, runs: pd.DataFrame, models: Sequence[str],
                        window: Sequence[str] = ANDREONI_WINDOW) -> pd.DataFrame:
    """Our limit in the Andreoni et al. (2020) window and over the full survey, next to the published limit.

    ``published_projected_full`` carries the published limit to the full survey with the <VT> growth we measure
    (assumes the ratio of the published efficiency to ours is the same in every epoch);
    ``published_duration_scaled`` scales it with survey duration only, for contrast.
    """
    from . import literature as L
    n_months = (pd.Period(window[1], "M") - pd.Period(window[0], "M")).n + 1
    dur = float(runs["survey_duration_years"].iloc[0])
    rows = []
    for m in models:
        if m not in ANDREONI_COUNTERPART:
            continue
        full = headline_limits(runs, [m]).iloc[0]
        win = vt_in_window(tm, runs, m, *window)
        ref = L.lookup(ANDREONI_COUNTERPART[m])
        gain = full.vt_gpc3_yr / win["vt"]
        rows.append(dict(model=m, published_key=ref["key"], published=ref["limit"], published_model=ref["model"],
                         vt_window=win["vt"], r95_window=win["r95"], vt_full=full.vt_gpc3_yr, r95_full=full.r95,
                         vt_gain_full_over_window=gain, duration_gain=dur / (n_months / 12),
                         window_change_vs_published=win["r95"] / ref["limit"] - 1,
                         published_projected_full=ref["limit"] / gain,
                         published_duration_scaled=ref["limit"] * (n_months / 12) / dur))
    return pd.DataFrame(rows)


def criteria_response(s1: pd.DataFrame, fid: pd.DataFrame, fiducial: dict = FIDUCIAL_CRITERIA) -> pd.DataFrame:
    """Per model and criterion: <VT> range over the one-at-a-time slice and the log-elasticity at the fiducial point.

    ``elasticity`` is d ln<VT> / d ln(criterion) from the neighbouring grid values (NaN where the fiducial value is
    0 or at the edge of the grid).
    """
    rows = []
    fid = fid if "model" not in fid.columns else fid.set_index("model")
    for m in fid.index:
        for c, x0 in fiducial.items():
            d = s1[(s1.varied == c) & (s1.model == m)].sort_values(c)
            if c not in d or d[c].nunique() < 2:
                continue
            v = d.effective_vt_gpc3_yr.to_numpy(); xv = d[c].to_numpy(float)
            i0 = int(np.argmin(np.abs(xv - x0)))
            e = np.nan
            if x0 > 0 and 0 < i0 < len(xv) - 1 and xv[i0 - 1] > 0:
                e = (np.log(v[i0 + 1]) - np.log(v[i0 - 1])) / (np.log(xv[i0 + 1]) - np.log(xv[i0 - 1]))
            rows.append(dict(model=m, criterion=c, vt_max_over_min=v.max() / v.min(),
                             vt_rel_min=v.min() / fid.loc[m, "effective_vt_gpc3_yr"],
                             vt_rel_max=v.max() / fid.loc[m, "effective_vt_gpc3_yr"], elasticity=e))
    return pd.DataFrame(rows)


def inclination_crosscheck(inc: pd.DataFrame, lim: pd.DataFrame, iso_model: str, fixed_model: str,
                           iota_rad: float = 0.45) -> dict:
    """Isotropic <VT>(iota) interpolated to the fixed-angle model's iota, compared with that model's own <VT>."""
    tt = inc.sort_values("center")
    v_interp = float(np.interp(np.cos(iota_rad), tt.center, tt.vt))
    v_fixed = float(lim.loc[fixed_model, "vt_gpc3_yr"])
    v_avg = float(np.sum(inc.vt * (inc.hi_edge - inc.lo_edge)))
    return dict(vt_interp=v_interp, vt_fixed=v_fixed, rel_diff=v_fixed / v_interp - 1,
                vt_iso_from_bins=v_avg, vt_iso_pooled=float(lim.loc[iso_model, "vt_gpc3_yr"]))


def distance_quantiles(zh: pd.DataFrame, model: str, q=(0.5, 0.9, 0.99)) -> dict:
    """Luminosity distance [Mpc] below which a fraction ``q`` of the recovered injections lie."""
    h = zh[zh.model == model].groupby(["z_lo", "z_hi"], as_index=False)["n_detected"].sum().sort_values("z_lo")
    c = np.cumsum(h.n_detected) / h.n_detected.sum()
    return {f"dL{int(round(p * 100))}": float(luminosity_distance_mpc(float(np.interp(p, c, h.z_hi)))[0]) for p in q}

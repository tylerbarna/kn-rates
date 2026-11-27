"""Every figure used by the notebooks, one function each.

Each function takes snapshot tables (usually via :class:`knrates.notebook.Context`) and returns the matplotlib
figure, so the paper notebook and the analysis notebooks draw identical figures from one definition.
Paper figures are marked ``(paper Fig. N)``.  ``literature`` arguments are lists of keys from
:mod:`knrates.literature` (``L.keys()`` lists them).
"""

from __future__ import annotations

import json
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analysis as A
from . import literature as L
from . import plotting as P


def _short(m: str) -> str:
    return P.SHORT_LABEL[m] + P.tag(m)


# ---------------------------------------------------------------------------------------------------- main result
def rate_limits(lim: pd.DataFrame, models: Sequence[str],
                literature: Sequence[str] = ("andreoni2020_gw170817", "andreoni2020_possis_isotropic",
                                             "andreoni2021_ztfrest_combined", "kasliwal2017_iptf", "gwtc4_bns_rate")):
    """95% limit per model (90–99% as a line) against published optical limits and the GW BNS rate (paper Fig. 3)."""
    selected = L.select(list(literature))
    rows = [(m, lim.loc[m]) for m in models]
    fig, ax = plt.subplots(figsize=(P.PAGE_WIDTH, 0.9 + 0.4 * len(rows)))
    y = np.arange(len(rows))[::-1]
    for yi, (m, r) in zip(y, rows):
        st = P.style(m)
        ax.errorbar(r.r95, yi, xerr=r.r95_err_mc, fmt=st["marker"], color=st["color"], ms=5)
        ax.plot([r.r90, r.r99], [yi, yi], color=st["color"], lw=0.8, alpha=0.6)
        ax.annotate(f"{r.r95:.0f}", (r.r99, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7)
    styles = P.lit_line_styles(); bands = P.lit_bands()
    for r in selected:
        if "limit" in r:
            cl = "CL n/s" if r["cl"].startswith("not stated") else r["cl"].replace(" (implied)", "")
            ax.axvline(r["limit"], lw=0.9, **next(styles), label=f"{r['ref'].replace(' et al.', '+')} ({r['model']}; {cl})")
        else:
            name = r["ref"].split(" population")[0].replace("LVK ", "")
            ax.axvspan(r["low"], r["high"], lw=0, zorder=0, label=f"{name} BNS (90%)", **next(bands))
    ax.set_yticks(y, [P.style(m)["label"] for m, _ in rows]); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(8, max([6e3, *[r.get("limit", r.get("high", 0)) * 1.5 for r in selected]]))
    ax.set_xlabel(P.RATE_LABEL + r"  (marker: 95%; line: 90–99%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=6.5, borderaxespad=0)
    return fig


def distance_reach(zh: pd.DataFrame, runs: pd.DataFrame, models: Sequence[str]):
    """Recovery efficiency and cumulative distance distribution of recovered injections (paper Fig. 4)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(P.PAGE_WIDTH, 2.6))
    for m in models:
        st = P.style(m)
        e = A.efficiency_vs_redshift(zh, runs, m); e = e[e.n_detected > 0]
        ax1.plot(A.luminosity_distance_mpc(e.z_center.to_numpy()), e.efficiency, color=st["color"], lw=1, label=_short(m))
        h = zh[zh.model == m].groupby(["z_lo", "z_hi"], as_index=False)["n_detected"].sum().sort_values("z_lo")
        c = np.cumsum(h.n_detected) / h.n_detected.sum()
        ax2.plot(A.luminosity_distance_mpc(h.z_hi.to_numpy()), c, color=st["color"], label=_short(m))
    ax1.set_yscale("log"); ax1.set_ylim(1e-5, 1); ax1.set_xlim(0, 700)
    ax1.set_xlabel("Luminosity distance [Mpc]"); ax1.set_ylabel(r"Recovery efficiency $\epsilon(d_L)$")
    ax2.set_xlim(0, 700); ax2.set_xlabel("Luminosity distance [Mpc]"); ax2.set_ylabel("Cumulative fraction of recoveries")
    ax2.legend(loc="lower right", fontsize=6.5)
    return fig


# ---------------------------------------------------------------------------------------------- time and history
def sensitivity_timeline(tm: pd.DataFrame, runs: pd.DataFrame, models: Sequence[str],
                         literature: Sequence[str] = ("andreoni2020_gw170817", "andreoni2020_possis_isotropic")):
    """Monthly sensitivity and the cumulative limit versus survey end date, with published ZTF limits placed at the
    end of their data and carried forward with the measured <VT> growth (paper Fig. 1)."""
    selected = L.select(list(literature), "rate_limits")
    counterpart = {v: k for k, v in A.ANDREONI_COUNTERPART.items() if k in models}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(P.COL_WIDTH, 3.9), sharex=True, height_ratios=[1, 1.6])
    e0 = A.vt_by_epoch(tm, runs, models[0])
    ax1.step(e0["month"].dt.to_timestamp(), e0["vt_month"] / e0["vt_month"].mean(), where="post", color="0.25", lw=0.9)
    ax1.set_ylabel("Relative monthly\n" + r"$\langle VT\rangle$")
    curves = {}
    for m in models:
        e = A.vt_by_epoch(tm, runs, m); curves[m] = e = e[e.vt_cumulative > 0]
        ax2.plot(e["month_end"], e["r95_cumulative"], color=P.style(m)["color"], label=_short(m))
    markers = P.lit_markers(); carried = False
    for r in selected:
        t_end = pd.Timestamp(r["data_end"]) + pd.Timedelta(days=1)   # exclusive: month_end is 23:59:59.999 of the last day
        ax2.plot(t_end, r["limit"], next(markers), color="k", ms=6, label=f"{r['ref'].replace(' et al. ', '+')}: {r['model']}")
        m = counterpart.get(r["key"])
        if m in curves:
            e = curves[m]; at = e[e["month_end"] < t_end].iloc[-1]
            after = e[e["month_end"] >= at["month_end"]]
            ax2.plot(after["month_end"], r["limit"] * at["vt_cumulative"] / after["vt_cumulative"], "--", lw=0.8,
                     color=P.style(m)["color"]); carried = True
            ax2.annotate(f"{100 * (at['r95_cumulative'] / r['limit'] - 1):+.0f}%", (t_end, r["limit"]),
                         xytext=(-5, 0), textcoords="offset points", ha="right", va="center", fontsize=6,
                         bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    if carried:
        ax2.plot([], [], "k--", lw=0.8, label="carried forward")
    ax2.set_yscale("log"); ax2.set_ylim(15, 3e4)
    ax2.set_ylabel(P.RATE_LABEL); ax2.set_xlabel("Survey end date")
    ax2.legend(loc="lower left", fontsize=6.3)   # the lower left is empty; elsewhere it hides the markers
    return fig


def andreoni_window(cmp: pd.DataFrame, literature_lines: Sequence[str] = ("andreoni2021_ztfrest_combined",
                                                                           "kasliwal2017_iptf")):
    """Our limits next to Andreoni et al. (2020): in their 23-month window, and carried to the full survey."""
    grey = dict(color="0.78", edgecolor="0.3", lw=0.6)

    def bracket(ax, x0, x1, ours, theirs, clear=0.0):
        top = max(ours, theirs, clear) * 1.12
        ax.plot([x0, x0, x1, x1], [ours * 1.04, top, top, theirs * 1.04], color="k", lw=0.6)
        ax.text((x0 + x1) / 2, top * 1.04, f"{100 * (ours / theirs - 1):+.0f}%", ha="center", va="bottom", fontsize=7)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(P.PAGE_WIDTH, 3.1), sharey=True)
    xx = np.arange(len(cmp)); w = 0.38
    cols = [P.style(m)["color"] for m in cmp.model]
    ax1.bar(xx - w / 2, cmp.r95_window, w, color=cols, label="This work")
    ax1.bar(xx + w / 2, cmp.published, w, **grey, label="Andreoni et al. (2020), published")
    ax2.bar(xx - w / 2, cmp.r95_full, w, color=cols)
    ax2.bar(xx + w / 2, cmp.published_projected_full, w, **grey, hatch="///",
            label="Andreoni et al. (2020) carried to 8 yr\nwith the measured ZTF $\\langle VT\\rangle$ growth")
    ax2.plot(xx + w / 2, cmp.published_duration_scaled, "_", color="k", ms=11, mew=1.2,
             label="duration scaling only (for contrast)")
    for i, r in cmp.reset_index(drop=True).iterrows():
        bracket(ax1, i - w / 2, i + w / 2, r.r95_window, r.published)
        bracket(ax2, i - w / 2, i + w / 2, r.r95_full, r.published_projected_full, clear=r.published_duration_scaled)
    lines = L.select(list(literature_lines), "rate_limits")
    for rr, sty in zip(lines, P.lit_line_styles()):
        ax2.axhline(rr["limit"], lw=0.9, **sty, label=f"{rr['ref']}: {rr['model']}")
    for ax in (ax1, ax2):
        ax.set_xticks(xx, [_short(m) for m in cmp.model], fontsize=7.5)
    ax1.set_title("First 23 months of ZTF", fontsize=9); ax2.set_title("Full 8-year survey", fontsize=9)
    ax1.set_ylabel(P.RATE_LABEL); ax2.set_yscale("log")
    ax2.set_ylim(top=max([*cmp.r95_window, *cmp.published, *[r["limit"] for r in lines], 100]) * 2.2)
    P.plain_log(ax2, "y")
    ax1.legend(loc="upper left", fontsize=6.5, frameon=False); ax2.legend(loc="upper left", fontsize=6, frameon=False)
    return fig


# ---------------------------------------------------------------------------------------------------- viewing angle
def viewing_angle(src: pd.DataFrame, runs: pd.DataFrame, lim: pd.DataFrame, lc: Optional[pd.DataFrame],
                  literature: Sequence[str] = ("mooley2022_gw170817_viewing_angle",), n_bins: int = 20):
    """(a) model light curves vs. iota, (b) limit at fixed iota, (c) share of isotropic <VT> per degree,
    (d) distance reach by orientation (paper Fig. 2)."""
    selected = L.select(list(literature), "other")
    iso = [m for m in ("Bu2026Vary", "GW170817_Bu2026_Isotropic") if m in set(src.model)]
    inc = {m: A.inclination_efficiency(src, runs, m, n_bins=n_bins) for m in iso}
    fig, axs = plt.subplots(2, 2, figsize=(P.PAGE_WIDTH, 4.6))
    (axa, axb), (axc, axd) = axs
    cmap = plt.get_cmap("viridis")
    if lc is not None and not lc.empty:
        s0 = lc[(lc.param_set == "suite01") & (lc.band == "g")]
        incs = sorted(s0.inclination_deg.unique())
        for k, ii in enumerate(incs):
            d = s0[s0.inclination_deg == ii].sort_values("t_days")
            axa.plot(d.t_days, d.abs_mag, color=cmap(k / max(1, len(incs) - 1)), lw=1, label=f"{ii:g}°")
        axa.set_xscale("log"); axa.set_xlim(0.19, 10); axa.set_ylim(-11.5, -17.0)
        axa.legend(loc="upper right", fontsize=5.5, ncol=2, title=r"$\iota$", title_fontsize=6)
    axa.set_xlabel("Rest-frame time [d]"); axa.set_ylabel(r"$M_g$ [AB]"); axa.set_title("(a) Bu2026 model, ZTF $g$", fontsize=8)
    for m, t in inc.items():
        st = P.style(m)
        xerr = [t.iota_deg_center - t.iota_deg_lo, t.iota_deg_hi - t.iota_deg_center]
        axb.errorbar(t.iota_deg_center, t.r95, xerr=xerr, fmt=st["marker"], color=st["color"], ms=2.5, lw=0.7, label=_short(m))
        tt = t.sort_values("iota_deg_lo"); w = tt.vt * (tt.hi_edge - tt.lo_edge)
        edges = np.append(tt.iota_deg_lo.to_numpy(), tt.iota_deg_hi.to_numpy()[-1])
        axc.stairs((w / w.sum()).to_numpy() / (tt.iota_deg_hi - tt.iota_deg_lo).to_numpy(), edges, color=st["color"], label=_short(m))
    for f in ("Bu2026Fixed", "GW170817_Bu2026_FixedView"):
        if f in lim.index:
            st = P.style(f)
            axb.plot(np.degrees(0.45), lim.loc[f, "r95"], st["marker"], mfc="none", mec=st["color"], ms=6, mew=1.1, label=_short(f))
    for ref, shade in zip(selected, ["0.88", "0.94"]):
        axb.axvspan(ref["low"], ref["high"], color=shade, zorder=0)
    axb.set_yscale("log"); P.plain_log(axb, "y"); axb.set_xlim(0, 90); axb.set_xlabel(r"Viewing angle $\iota$ [deg]")
    axb.set_ylabel(P.RATE_LABEL); axb.legend(loc="upper left", fontsize=5.5); axb.set_title(r"(b) Limit at fixed $\iota$", fontsize=8)
    th = np.linspace(0, 90, 181)
    axc.plot(th, np.sin(np.radians(th)) * np.pi / 180, "k--", lw=0.9, label=r"$\sin\iota$ (all mergers)")
    axc.set_xlim(0, 90); axc.set_xlabel(r"Viewing angle $\iota$ [deg]"); axc.set_ylabel(r"Share of isotropic $\langle VT\rangle$ per deg")
    axc.legend(loc="upper right", fontsize=5.5); axc.set_title("(c) Which orientations are searched", fontsize=8)
    if iso:
        s_ = src[src.model == iso[0]]
        edges = np.radians([0, 30, 45, 60, 75, 90])
        for i in range(len(edges) - 1):
            sel = s_[(s_.inclination_rad >= edges[i]) & (s_.inclination_rad < edges[i + 1])]
            d = np.sort(A.luminosity_distance_mpc(sel.z.to_numpy()))
            axd.plot(d, np.arange(1, len(d) + 1) / len(d), color=cmap(i / (len(edges) - 2)),
                     label=f"{np.degrees(edges[i]):.0f}–{np.degrees(edges[i + 1]):.0f}°")
    axd.set_xlabel("Luminosity distance of recovered KNe [Mpc]"); axd.set_ylabel("Cumulative fraction")
    axd.legend(loc="lower right", fontsize=5.5, title=r"$\iota$", title_fontsize=6); axd.set_title("(d) Distance reach by orientation", fontsize=8)
    return fig


# ------------------------------------------------------------------------------------------------------ systematics
def convergence(conv: pd.DataFrame, lim: pd.DataFrame, models: Sequence[str]):
    """Scatter of <VT> versus number of injections (sub-sampled from the real realisations) and the probability of
    zero recoveries (paper Fig. 5)."""
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(P.PAGE_WIDTH, 2.5))
    nn = np.logspace(2, 5, 200)
    for m in models:
        c = conv[conv.model == m]; st = P.style(m)
        ax.plot(c.n_injected, 100 * c.rel_scatter_vt, st["marker"], color=st["color"], ms=3.5, label=_short(m))
        ax.plot(c.n_injected, 100 * c.rel_err_binomial, "-", color=st["color"], lw=0.8)
        axz.plot(nn, np.exp(-lim.loc[m, "efficiency"] * nn), color=st["color"], lw=1, label=_short(m))
        ok = c.p_zero_empirical > 0
        axz.plot(c.n_injected[ok], c.p_zero_empirical[ok], st["marker"], color=st["color"], ms=3.5)
    ax.set_xscale("log"); ax.set_yscale("log"); P.plain_log(ax, "y")
    ax.set_xlabel("Injected transients $N$"); ax.set_ylabel(r"Scatter of $\langle VT\rangle$ [%]")
    ax.legend(loc="lower left", fontsize=6.3)
    ax.text(0.97, 0.95, r"lines: $(N\epsilon)^{-1/2}$", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    axz.set_xscale("log"); axz.set_yscale("log"); axz.set_ylim(1e-4, 1.2); axz.set_xlim(1e2, 1e5)
    axz.set_xlabel("Injected transients $N$"); axz.set_ylabel(r"$P(N_{\rm rec}=0)$")
    return fig


def zmax_check(zh: pd.DataFrame, runs: pd.DataFrame, models: Sequence[str], n_per_run: float = 1e6):
    """Fraction of <VT> recovered below a trial injection boundary, and the MC error per 10^6 injections it implies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(P.PAGE_WIDTH, 2.6))
    zz = np.linspace(0.02, 0.30, 57)
    for m in models:
        st = P.style(m)
        cvt = A.cumulative_vt_vs_redshift(zh, runs, m)
        ax1.plot(cvt.z_hi, cvt.vt_fraction_below, color=st["color"], label=_short(m))
        r = runs[runs.model == m]; z0 = r.z_max.iloc[0]
        h = zh[zh.model == m].groupby(["z_lo", "z_hi"], as_index=False)["n_detected"].sum()
        v0 = A.comoving_volume_gpc3(z0)[0]; k = h.n_detected.sum() / r.n_simulated.sum()
        noise = []
        for z in zz:
            frac = h.loc[h.z_hi <= z + 1e-9, "n_detected"].sum() / h.n_detected.sum() if z < z0 else 1.0
            eff = k * frac * v0 / A.comoving_volume_gpc3(z)[0]
            noise.append(100 / np.sqrt(n_per_run * eff) if eff > 0 else np.nan)
        ax2.plot(zz, noise, color=st["color"], label=_short(m))
    z0 = runs.z_max.iloc[0]
    for ax in (ax1, ax2):
        ax.axvline(z0, color="0.5", ls=":", lw=1); ax.set_xlabel(r"Injection boundary $z_{\max}$")
    ax1.set_ylabel(r"Fraction of $\langle VT\rangle$ below $z_{\max}$"); ax1.set_xlim(0, 0.2)
    ax2.set_ylabel(r"MC error per $10^6$ injections [%]"); ax2.set_yscale("log"); P.plain_log(ax2, "y")
    ax1.legend(loc="lower right", fontsize=7)
    return fig


def at2017gfo_lightcurves(lc: pd.DataFrame, include_metzger: bool = False):
    """Model light curves against AT2017gfo photometry; returns (fig, mean model-minus-data residual table)."""
    data = lc[lc.param_set == "AT2017gfo_PS1"]
    curves = {"suite01": (r"Bu2026, $\iota=25.8^\circ$", P.style("Bu2026Fixed")["color"], "-")}
    if include_metzger:
        curves.update({
            "metzger_template_k398": (r"Metzger, $\kappa=398$", P.style("Metzger")["color"], "--"),
            "metzger_blue_k0.5": (r"Metzger blue, $\kappa=0.5$", P.style("GW170817_Metzger_Blue")["color"], "-."),
            "metzger_mean_k1": (r"Metzger mean, $\kappa=1$", P.style("GW170817_Metzger_Mean")["color"], ":")})
    fig, axes = plt.subplots(1, 3, figsize=(P.PAGE_WIDTH, 2.6), sharey=True)
    rows = []
    for ax, band in zip(axes, "gri"):
        d = data[data.band == band]
        ax.errorbar(d.t_days, d.abs_mag, yerr=d.mag_err, fmt="o", color="k", ms=3, lw=0.8, label="AT2017gfo (PS1)")
        for key, (lab, col, ls) in curves.items():
            c = lc[(lc.param_set == key) & (lc.band == band)]
            if key == "suite01":
                c = c[c.inclination_deg == 25.8]
            c = c.sort_values("t_days")
            if c.empty:
                continue
            ax.plot(c.t_days, c.abs_mag, color=col, ls=ls, lw=1.2, label=lab)
            dd = d[(d.t_days >= c.t_days.min()) & (d.t_days <= 7.5)]
            res = np.interp(np.log(dd.t_days), np.log(c.t_days), c.abs_mag) - dd.abs_mag
            for lo, hi in [(0, 2), (2, 4), (4, 7.5)]:
                sel = (dd.t_days > lo) & (dd.t_days <= hi)
                rows.append(dict(model=lab, band=band, phase=f"{lo}-{hi} d", mean_model_minus_data=res[sel].mean()))
        ax.set_xscale("log"); ax.set_xlim(0.3, 10); ax.set_ylim(-9.5, -17.5)
        ax.set_xlabel("Time since merger [d]"); ax.set_title(f"{band} band", fontsize=9)
    axes[0].set_ylabel("Absolute magnitude [AB]")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=7, frameon=False)
    res = pd.DataFrame(rows).pivot_table(index=["model", "band"], columns="phase", values="mean_model_minus_data")
    return fig, res


def population_regimes(lim3: pd.DataFrame, regimes: Sequence[str], populations: Sequence[str]):
    """Suite 03: <VT> and limit per selection regime for the AT2017gfo-like models and the broad population."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(P.PAGE_WIDTH, 2.7))
    xx = np.arange(len(regimes)); w = 0.8 / max(1, len(populations))
    pal = [P.OKABE_ITO[c] for c in ("blue", "vermillion", "green", "purple", "orange")]
    for j, p in enumerate(populations):
        d = lim3[lim3.population_type == p].set_index("criteria_regime").reindex(regimes)
        off = (j - (len(populations) - 1) / 2) * w
        ax1.bar(xx + off, d.vt_gpc3_yr * 1e3, width=w, color=pal[j % len(pal)], label=p.replace("_", " "),
                yerr=d.vt_err_empirical * 1e3, capsize=2)
        ax2.bar(xx + off, d.r95, width=w, color=pal[j % len(pal)], yerr=d.r95_err_mc, capsize=2)
    for ax in (ax1, ax2):
        ax.set_xticks(xx, [r.replace("_", "\n") for r in regimes]); ax.set_yscale("log"); P.plain_log(ax, "y")
    ax1.set_ylabel(r"$\langle VT\rangle$ [$10^{-3}$ Gpc$^3$ yr]"); ax2.set_ylabel(P.RATE_LABEL)
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               fontsize=7.5, frameon=False)
    return fig


# --------------------------------------------------------------------------------------------- selection criteria
FIG6_CRITERIA = ("snr_threshold", "snr_threshold_secondary", "min_detections", "min_time_separation_hours",
                 "min_fade_rate", "min_rise_rate")


def criteria_one_at_a_time(s1: pd.DataFrame, fid: pd.DataFrame, criteria: Sequence[str] = FIG6_CRITERIA, ncol: int = 3):
    """<VT> relative to the fiducial search when one criterion is varied (paper Fig. 6 with the default criteria)."""
    fid = fid if "model" not in fid.columns else fid.set_index("model")
    criteria = [c for c in criteria if s1.loc[s1.varied == c, c].nunique() > 1] if len(s1) else []
    nrow = int(np.ceil(len(criteria) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(P.PAGE_WIDTH, 1.8 * nrow), sharey=True, squeeze=False)
    for ax, c in zip(axes.flat, criteria):
        for m in fid.index:
            d = s1[(s1.varied == c) & (s1.model == m)].sort_values(c); st = P.style(m)
            ax.plot(d[c], d.effective_vt_gpc3_yr / fid.loc[m, "effective_vt_gpc3_yr"], marker=st["marker"], ms=2.5,
                    color=st["color"], label=_short(m))
        ax.axvline(A.FIDUCIAL_CRITERIA[c], color="0.5", ls=":", lw=0.8); ax.axhline(1, color="0.85", lw=0.6, zorder=0)
        ax.set_xlabel(P.CRITERION_LABEL[c])
        if c in ("min_detections", "min_detections_primary"):
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)); ax.xaxis.set_minor_locator(plt.NullLocator())
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\langle VT\rangle/\langle VT\rangle_{\rm fid}$")
    for ax in axes.flat[len(criteria):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="lower left", fontsize=6)
    return fig


def criteria_ranking(resp: pd.DataFrame):
    """Total <VT> range over the tested values of each criterion (others at fiducial)."""
    tab = resp.pivot(index="criterion", columns="model", values="vt_max_over_min")
    tab = tab.loc[tab.mean(axis=1).sort_values().index]
    models = list(tab.columns)
    fig, ax = plt.subplots(figsize=(P.PAGE_WIDTH * 0.7, 0.3 * len(tab) + 0.9))
    yy = np.arange(len(tab)); h = 0.8 / len(models)
    for j, m in enumerate(models):
        ax.barh(yy + (j - (len(models) - 1) / 2) * h, tab[m] - 1, height=h, left=1, color=P.style(m)["color"], label=_short(m))
    ax.set_yticks(yy, [P.CRITERION_LABEL[c] for c in tab.index])
    ax.set_xlabel(r"$\langle VT\rangle_{\max}/\langle VT\rangle_{\min}$ over the tested range")
    ax.legend(loc="lower right", fontsize=7)
    return fig


def criteria_pairs(s2: pd.DataFrame, fid: pd.DataFrame, model: str,
                   pairs: Sequence = (("snr_threshold", "snr_threshold_secondary"), ("min_detections", "max_timespan_days"),
                                      ("min_rise_rate", "min_fade_rate"), ("min_detections", "min_fade_rate"))):
    """<VT>/<VT>_fid on two-criterion grids for one model (blank cells: combinations excluded as invalid)."""
    fid = fid if "model" not in fid.columns else fid.set_index("model")
    nc = 2; nr = int(np.ceil(len(pairs) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(P.PAGE_WIDTH, 2.6 * nr), squeeze=False)
    for ax, (a, b) in zip(axes.flat, pairs):
        d = s2[(s2.model == model) & (((s2.varied_a == a) & (s2.varied_b == b)) | ((s2.varied_a == b) & (s2.varied_b == a)))]
        piv = d.pivot_table(index=b, columns=a, values="effective_vt_gpc3_yr") / fid.loc[model, "effective_vt_gpc3_yr"]
        arr = piv.to_numpy()
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(piv.shape[1]), [f"{v:g}" for v in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f"{v:g}" for v in piv.index])
        ax.set_xlabel(P.CRITERION_LABEL[a]); ax.set_ylabel(P.CRITERION_LABEL[b])
        ax.minorticks_off(); ax.tick_params(top=False, right=False)
        if arr.size <= 80:
            for (i, j), val in np.ndenumerate(arr):
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if val < np.nanmean(arr) else "black")
        fig.colorbar(im, ax=ax, label=r"$\langle VT\rangle/\langle VT\rangle_{\rm fid}$", shrink=0.9)
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)
    fig.suptitle(P.SHORT_LABEL[model], fontsize=9)
    return fig


def named_regimes(nc: pd.DataFrame, order: Sequence[str] = ("Ultra-relaxed", "Relaxed (S03-like)", "Fiducial",
                                                             "N_min = 3", "Strict (S03-like)", "Ultra-strict")):
    """95% limit for named search strategies, from the most permissive to the strictest."""
    order = [r for r in order if r in set(nc.regime)]
    tab = nc.pivot(index="regime", columns="model", values="rate_upper_95").loc[order]
    fig, ax = plt.subplots(figsize=(P.PAGE_WIDTH * 0.75, 2.6))
    xx = np.arange(len(order)); w = 0.8 / len(tab.columns)
    for j, m in enumerate(tab.columns):
        ax.bar(xx + (j - (len(tab.columns) - 1) / 2) * w, tab[m], width=w, color=P.style(m)["color"], label=_short(m))
    ax.set_xticks(xx, [r.replace(" (S03-like)", "") for r in order], rotation=15)
    ax.set_yscale("log"); P.plain_log(ax, "y"); ax.set_ylabel(P.RATE_LABEL); ax.legend(loc="upper left", fontsize=7)
    return fig


# --------------------------------------------------------------------------------------------------------- table 1
def table1(lim: pd.DataFrame, runs: pd.DataFrame, models: Sequence[str]):
    """Table 1 of the paper: returns (csv-ready DataFrame, LaTeX-ready DataFrame, column format)."""
    def desc(m):
        p = {k.lower(): v for k, v in json.loads(runs.loc[runs.model == m, "params_json"].iloc[0]).items()}
        if "kappa" in p:
            return f"$M_{{\\rm ej}}={p['mej']:g}\\,M_\\odot$, $v={p['vej']:g}c$, $\\kappa={p['kappa']:g}$"
        inc = p.get("inclination_em")
        inc_s = ("isotropic" if (p.get("vary_inclination") is True or inc == "varied")
                 else f"$\\iota={np.degrees(float(inc)):.1f}^\\circ$")
        return f"$\\log M_{{\\rm dyn}}={p['log10_mej_dyn']:g}$, $\\log M_{{\\rm wind}}={p['log10_mej_wind']:g}$, {inc_s}"
    tab = lim.loc[list(models), ["n_simulated", "n_detected", "vt_gpc3_yr", "vt_err_empirical", "r90", "r95", "r99"]].reset_index()
    tab.insert(1, "suite", [P.SUITE_OF.get(m, "")[1:] for m in tab.model])
    tab.insert(2, "parameters", [desc(m) for m in tab.model])
    tex = tab.assign(vt=[f"{v * 1e3:.2f} $\\pm$ {e * 1e3:.2f}" for v, e in zip(tab.vt_gpc3_yr, tab.vt_err_empirical)],
                     label=[P.MODEL_STYLE[m]["label"] for m in tab.model])
    tex = tex[["label", "parameters", "n_detected", "vt", "r90", "r95", "r99"]]
    tex.columns = ["Model", "Parameters", "$N_{\\rm rec}$", "$\\langle VT\\rangle$ [$10^{-3}$ Gpc$^3$ yr]",
                   "$\\mathcal{R}_{90}$", "$\\mathcal{R}_{95}$", "$\\mathcal{R}_{99}$"]
    return tab, tex, "llrrrrr"


# --------------------------------------------------------------------------------------------- archival search (knsearch)
_STAGE_A_CUT_LABELS = {
    "all objects": "All objects",
    "pass_n_alerts_span": r"$\geq 2$ alerts in $\leq 14$ d",
    "pass_gal_lat": r"$|b| \geq 15^\circ$",
    "pass_real_bogus": r"Real--bogus ($\geq 2$ pass)",
    "pass_not_star": r"Not PS1 star ($1.5''$)",
    "pass_not_bright_star": r"Not Gaia $G<14$ ($20''$)",
    "pass_not_solar_system": r"Not asteroid ($10''$)",
    "pass_no_old_history": r"No prior history ($>30$ d)",
}


def stage_a_cutflow(cutflow_df: pd.DataFrame, run_ids: Optional[Sequence[str]] = None):
    """Sequential object survival across Stage A preselection cuts from ``knsearch_results_latest.zip``."""
    runs = list(run_ids) if run_ids is not None else sorted(cutflow_df["run_id"].unique())
    pal = [P.OKABE_ITO[k] for k in ("blue", "vermillion", "green", "purple", "orange", "skyblue", "black", "grey")]
    fig, ax = plt.subplots(figsize=(P.PAGE_WIDTH, 2.8))
    for idx, rid in enumerate(runs):
        sub = cutflow_df[cutflow_df["run_id"] == rid].sort_values("step_index")
        if sub.empty:
            continue
        labels = [_STAGE_A_CUT_LABELS.get(str(c), str(c)) for c in sub["cut"]]
        ax.plot(range(len(sub)), sub["n_remaining"], marker="o", ms=4, lw=1.2,
                color=pal[idx % len(pal)], label=rid.replace("msi/", ""))
        if idx == 0:
            ax.set_xticks(range(len(sub)), labels, rotation=22, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("Objects remaining")
    ax.legend(loc="upper right", ncol=2, fontsize=6.5, frameon=False)
    return fig


def rb_calibration_summary(rb_df: pd.DataFrame):
    """Acceptance-matched random-forest ``rb`` fallback thresholds across ZTF classifier epochs."""
    fig, ax = plt.subplots(figsize=(P.COL_WIDTH * 1.35, 2.3))
    yy = np.arange(len(rb_df))[::-1]
    for yi, (_, r) in zip(yy, rb_df.iterrows()):
        th = float(r["threshold"])
        is_anchor = str(r.get("reference_model", "")) == "drb>=0.5"
        lo = float(r["interval_lo"]) if (not is_anchor and pd.notna(r.get("interval_lo"))) else th
        hi = float(r["interval_hi"]) if (not is_anchor and pd.notna(r.get("interval_hi"))) else th
        if lo <= th <= hi and hi > lo:
            ax.errorbar(
                th, yi,
                xerr=[[th - lo], [hi - th]],
                fmt="o", color=P.OKABE_ITO["blue"], ms=5, capsize=2.5,
            )
            label_x = hi
        else:
            ax.plot(th, yi, "o", color=P.OKABE_ITO["vermillion"], ms=5)
            label_x = th
        ax.annotate(f"{th:.3f}", (label_x, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7)
    ax.set_yticks(yy, [f"{r['model']} (ref: {r['reference_model']})" for _, r in rb_df.iterrows()], fontsize=7.5)
    ax.set_xlim(0.15, 0.70)
    ax.set_xlabel(r"Calibrated $\mathtt{rb}$ threshold")
    return fig


# ----------------------------------------------------------------------- companion paper II (suite 09 imposters)
_REGIME_SHORT = {
    "No_Fade": "No fade",
    "Slow_Fade_01": "Fade 0.1",
    "Fiducial_PureFade": "Fiducial\n(pure fade)",
    "Fiducial_Rise0_Suite01": "Fiducial\n(rise=0)",
    "Fast_Fade_05": "Fade 0.5",
    "VeryFast_Fade_08": "Fade 0.8",
    "AlertOnly_SNR5": r"Alert $5\sigma$",
    "ThreeDet_Fade03": r"$N_{\min}=3$",
    "Loose_NoFade": "Loose",
    "Relaxed_Fade01": "Relaxed",
    "Moderate_Fade02": "Mod. 0.2",
    "Fiducial": "Fiducial",
    "ZTFReST_2Prim": "2-alert+FP",
    "Andreoni2020_Alert5": r"2$\times 5\sigma$ 7d",
    "ShortSpan7d_Fade04": "7d f0.4",
    "Strict_Fade05": "Strict f0.5",
}


def wls_slope_systematic(df_9a: pd.DataFrame):
    """Noisy-vs.-noiseless WLS fade-slope ratio and pure-fade vs. ``min_rise_rate=0`` validation (Paper II Fig. 7)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(P.PAGE_WIDTH, 2.55), gridspec_kw={"width_ratios": [1.55, 1.0]})
    regimes = [r for r in df_9a["regime"].unique() if r != "Fiducial_Rise0_Suite01"]
    xx = np.arange(len(regimes))
    for m in ("Bu2026Fixed", "Bu2026Vary"):
        sub = df_9a[df_9a["model"] == m].set_index("regime").reindex(regimes)
        if sub.empty:
            continue
        st = P.style(m)
        ax1.plot(xx, sub["noisy_over_noiseless_ratio"], marker=st["marker"], color=st["color"], lw=1.2, label=_short(m))
    ax1.axhline(1.0, color="0.5", ls=":", lw=0.9)
    ax1.set_xticks(xx, [_REGIME_SHORT.get(r, r) for r in regimes], rotation=22, ha="right", fontsize=6.8)
    ax1.set_ylabel(r"$\langle VT\rangle_{\rm noisy} / \langle VT\rangle_{\rm noiseless}$")
    ax1.set_title("(a) Photometric-scatter WLS slope bias", fontsize=8.5)
    ax1.legend(loc="lower left", fontsize=6.5, frameon=False)

    # Panel (b): Fiducial pure-fade (min_rise_rate=99) vs Suite 01 rise=0, noiseless vs noisy
    models = [m for m in ("Bu2026Fixed", "Bu2026Vary") if m in set(df_9a["model"])]
    ym = np.arange(len(models))
    w = 0.24
    bars = [
        ("Engine pure fade", "Fiducial_PureFade", "engine_direct_noiseless_n", P.OKABE_ITO["blue"]),
        ("Engine rise=0 (S01)", "Fiducial_Rise0_Suite01", "engine_direct_noiseless_n", P.OKABE_ITO["skyblue"]),
        ("Filter noisy (S02)", "Fiducial_PureFade", "n_pass_noisy", P.OKABE_ITO["vermillion"]),
    ]
    for j, (lab, reg, col_n, col_c) in enumerate(bars):
        vals = []
        for m in models:
            r_pf = df_9a[(df_9a["model"] == m) & (df_9a["regime"] == "Fiducial_PureFade")].iloc[0]
            r_tg = df_9a[(df_9a["model"] == m) & (df_9a["regime"] == reg)].iloc[0]
            base = float(r_pf["n_pass_noiseless"])
            v = float(r_tg[col_n]) if pd.notna(r_tg[col_n]) else float(r_tg["n_pass_noiseless"])
            vals.append(v / base if base > 0 else np.nan)
        ax2.bar(ym + (j - 1) * w, vals, width=w, color=col_c, label=lab)
    ax2.axhline(1.0, color="0.5", ls=":", lw=0.9)
    ax2.set_xticks(ym, [_short(m) for m in models], fontsize=7.5)
    ax2.set_ylim(0.70, 1.15)
    ax2.set_ylabel("Counts / noiseless pure-fade filter")
    ax2.set_title(r"(b) Fiducial parity & $\mathtt{min\_rise\_rate}=0$", fontsize=8.5)
    ax2.legend(loc="lower right", fontsize=6.0, frameon=False)
    return fig


def imposter_frontier_and_colors(
    frontier_df: pd.DataFrame,
    color_df: pd.DataFrame,
    *,
    signal_model: str = "Bu2026Vary",
    model_variant: str = "hybrid_fiesta",
):
    """Contaminant yields, ``d(g-r)/dt`` color separation, and 2D Completeness-Purity Pareto frontier (Paper II Fig. 8)."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(P.PAGE_WIDTH, 2.75))

    # Panel (a): Expected 8-yr contaminant yields vs signal across regimes (no color gate)
    sub_f = frontier_df[
        (frontier_df["signal_model"] == signal_model)
        & (frontier_df["model_variant"] == model_variant)
        & (frontier_df["color_gate"] == "none")
    ].copy()
    reg_order = [r for r in (
        "Loose_NoFade", "Relaxed_Fade01", "Moderate_Fade02", "Fiducial",
        "ZTFReST_2Prim", "ThreeDet_Fade03", "ShortSpan7d_Fade04", "Strict_Fade05",
    ) if r in set(sub_f["regime"])]
    sub_f = sub_f.set_index("regime").reindex(reg_order).reset_index()
    xx = np.arange(len(sub_f))
    y_imp = sub_f["n_imposters_expected"].to_numpy(float)
    y_lo = np.maximum(0.0, y_imp - sub_f["n_imposters_lo"].to_numpy(float))
    y_hi = np.maximum(0.0, sub_f["n_imposters_hi"].to_numpy(float) - y_imp)
    ax1.errorbar(xx, np.maximum(y_imp, 1e-2), yerr=[y_lo, y_hi], fmt="s-", color=P.OKABE_ITO["vermillion"],
                 ms=3.8, lw=1.1, capsize=2, label="Extragalactic imposters")
    ax1.plot(xx, np.maximum(sub_f["n_signal_expected"].to_numpy(float), 1e-2), "o--",
             color=P.OKABE_ITO["blue"], ms=3.8, lw=1.1, label=r"KN ($\mathcal{R}=100$)")
    ax1.set_yscale("log")
    ax1.set_xticks(xx, [_REGIME_SHORT.get(r, r) for r in sub_f["regime"]], rotation=32, ha="right", fontsize=6.3)
    ax1.set_ylabel("Expected 8-yr ZTF yield (generic log)")
    ax1.set_title("(a) Yield vs. cut strictness", fontsize=8.3)
    ax1.legend(loc="upper right", fontsize=5.8, frameon=False)

    # Panel (b): d(g-r)/dt color cooling slopes at Fiducial
    cdf = color_df.copy()
    # Keep fiesta engine where available, else parametric
    cdf["prio"] = np.where(cdf["engine"] == "fiesta", 0, 1)
    cdf = cdf.sort_values(["prio", "median_dgr_dt"], ascending=[True, False]).drop_duplicates("transient_class")
    cdf = cdf[cdf["n_with_gr_slope"] >= 1].sort_values("median_dgr_dt")
    yy = np.arange(len(cdf))
    for yi, (_, r) in zip(yy, cdf.iterrows()):
        is_kn = str(r["transient_class"]).startswith("Bu2026")
        col = P.OKABE_ITO["blue"] if is_kn else P.OKABE_ITO["grey"]
        med = float(r["median_dgr_dt"])
        lo_e = max(0.0, med - float(r["p16_dgr_dt"])) if pd.notna(r["p16_dgr_dt"]) else 0.0
        hi_e = max(0.0, float(r["p84_dgr_dt"]) - med) if pd.notna(r["p84_dgr_dt"]) else 0.0
        ax2.errorbar(med, yi, xerr=[[lo_e], [hi_e]], fmt="o" if is_kn else "s", color=col, ms=4, lw=1.0, capsize=2)
    ax2.axvline(0.15, color=P.OKABE_ITO["green"], ls="--", lw=0.9, label=r"$+0.15$ mag d$^{-1}$")
    ax2.set_yticks(yy, [str(t).replace("Bu2026Fixed", r"KN $\iota=26^\circ$").replace("Bu2026Vary", "KN iso.")
                        for t in cdf["transient_class"]], fontsize=6.3)
    ax2.set_xlabel(r"$d(g-r)/dt$ [mag d$^{-1}$]")
    ax2.set_title("(b) Color reddening rate", fontsize=8.3)
    ax2.legend(loc="lower right", fontsize=5.8, frameon=False)

    # Panel (c): 2D Completeness-Purity Pareto frontier across regimes x color gates
    all_f = frontier_df[
        (frontier_df["signal_model"] == signal_model)
        & (frontier_df["model_variant"] == model_variant)
    ].copy()
    gate_style = {
        "none": ("o", P.OKABE_ITO["grey"], "No color gate"),
        "veto_flat_blue": ("^", P.OKABE_ITO["orange"], "Veto flat/blue"),
        "require_reddening_015": ("s", P.OKABE_ITO["green"], r"Reddening $\geq 0.15$"),
        "require_reddening_025": ("D", P.OKABE_ITO["purple"], r"Reddening $\geq 0.25$"),
    }
    for gname, (mk, col, lab) in gate_style.items():
        g = all_f[all_f["color_gate"] == gname]
        if g.empty:
            continue
        ax3.scatter(g["completeness_vs_fiducial"], 100.0 * g["purity"], marker=mk, s=18, color=col, alpha=0.85, label=lab)
    p_opt = all_f[all_f["pareto_optimal"]].sort_values("completeness_vs_fiducial")
    if not p_opt.empty:
        ax3.plot(p_opt["completeness_vs_fiducial"], 100.0 * p_opt["purity"], "k-", lw=1.1, zorder=4, label="Pareto frontier")
    ax3.axvline(1.0, color="0.6", ls=":", lw=0.8)
    ax3.set_yscale("log")
    P.plain_log(ax3, "y")
    ax3.set_xlabel(r"Completeness $\langle VT\rangle / \langle VT\rangle_{\rm fid}$")
    ax3.set_ylabel(r"Unvetted purity [%] ($\mathcal{R}_{\rm KN}=100$)")
    ax3.set_title("(c) Completeness--purity frontier", fontsize=8.3)
    ax3.legend(loc="upper right", fontsize=5.3, frameon=True)
    return fig

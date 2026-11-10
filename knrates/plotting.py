"""Consistent, publication-ready plotting style for all project notebooks.

Figures are sized for AAS journals (single column 3.35 in, double column 7.1 in),
use the colour-blind-safe Okabe-Ito palette, STIX (Times-like) math fonts and a
fixed colour/marker per model so the same model looks identical in every figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

COL_WIDTH = 3.35   # inches, AAS single column
PAGE_WIDTH = 7.1   # inches, AAS two-column span

OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#7F7F7F",
}

# Display labels, colours and markers per simulated model
MODEL_STYLE = {
    "Bu2026Fixed": dict(label=r"Bu2026, $\iota=26^\circ$ (S01)", color=OKABE_ITO["blue"], marker="o"),
    "Bu2026Vary": dict(label=r"Bu2026, isotropic $\iota$ (S01)", color=OKABE_ITO["vermillion"], marker="s"),
    "Metzger": dict(label=r"Metzger, $\kappa=398$ (S01)", color=OKABE_ITO["green"], marker="^"),
    "GW170817_Bu2026_FixedView": dict(label=r"Bu2026 GW170817, $\iota=26^\circ$ (S07)", color=OKABE_ITO["skyblue"], marker="o"),
    "GW170817_Bu2026_Isotropic": dict(label=r"Bu2026 GW170817, isotropic (S07)", color=OKABE_ITO["orange"], marker="s"),
    "GW170817_Metzger_Blue": dict(label=r"Metzger blue, $\kappa=0.5$ (S07)", color=OKABE_ITO["purple"], marker="D"),
    "GW170817_Metzger_Mean": dict(label=r"Metzger mean, $\kappa=1$ (S07)", color="#8C6D00", marker="v"),
}

SHORT_LABEL = {
    "Bu2026Fixed": r"Bu2026 $\iota=26^\circ$",
    "Bu2026Vary": r"Bu2026 iso.",
    "Metzger": r"Metzger $\kappa=398$",
    "GW170817_Bu2026_FixedView": r"Bu2026 $\iota=26^\circ$",
    "GW170817_Bu2026_Isotropic": r"Bu2026 iso.",
    "GW170817_Metzger_Blue": r"Metzger blue",
    "GW170817_Metzger_Mean": r"Metzger mean",
}

CRITERION_LABEL = {
    "snr_threshold": r"Primary S/N threshold",
    "snr_threshold_secondary": r"Secondary S/N threshold",
    "min_detections": r"Min. detections $N_{\rm min}$",
    "min_detections_primary": r"Min. primary detections",
    "max_timespan_days": r"Max. timespan [d]",
    "min_time_separation_hours": r"Min. separation [h]",
    "min_rise_rate": r"Min. rise rate [mag d$^{-1}$]",
    "min_fade_rate": r"Min. fade rate [mag d$^{-1}$]",
    "min_galactic_lat": r"Min. $|b|$ [deg]",
    "require_fast_transient": r"Require rate test",
}

RATE_LABEL = r"$\mathcal{R}_{95}$ [Gpc$^{-3}$ yr$^{-1}$]"
VT_LABEL = r"$\langle VT\rangle_{\rm eff}$ [Gpc$^{3}$ yr]"


def style(model: str) -> dict:
    return MODEL_STYLE.get(model, dict(label=model, color=OKABE_ITO["grey"], marker="o"))


def use_style(context: str = "paper") -> None:
    """Apply the project rcParams. ``context='notebook'`` enlarges fonts slightly."""
    base = 8.5 if context == "paper" else 10.5
    mpl.rcParams.update({
        "figure.dpi": 110 if context == "notebook" else 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": base - 1,
        "legend.title_fontsize": base - 1,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.75",
        "legend.fancybox": False,
        "axes.grid": False,
        "figure.constrained_layout.use": True,
        "axes.formatter.use_mathtext": True,
        "axes.titleweight": "normal",
    })


def savefig(fig, name: str, outdir: Optional[Path] = None, formats: Iterable[str] = ("pdf", "png")) -> list:
    """Save ``fig`` as ``<outdir>/<name>.<fmt>`` for each format; returns the paths written."""
    outdir = Path(outdir) if outdir is not None else Path("figures")
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = outdir / f"{name}.{fmt}"
        fig.savefig(p)
        paths.append(p)
    return paths


def fmt_sig(x: float, sig: int = 3) -> str:
    """Format a number to ``sig`` significant figures without scientific notation for moderate values."""
    if x == 0 or x != x:
        return str(x)
    from math import floor, log10

    digits = sig - int(floor(log10(abs(x)))) - 1
    digits = max(digits, 0)
    return f"{round(x, digits):,.{digits}f}"



def plain_log(ax, axis: str = "y", minor=(2, 3, 5)) -> None:
    """Label a log axis with plain numbers (e.g. 200, 300, 500) instead of 2x10^2-style labels."""
    import matplotlib.ticker as mt
    import numpy as np

    def fmt(v, _pos=None):
        if v <= 0:
            return ""
        if 1e-3 <= v < 1e5:
            return f"{v:g}"
        lg = np.log10(v)
        return f"$10^{{{int(round(lg))}}}$" if np.isclose(lg, round(lg)) else ""

    axes = ([ax.xaxis] if axis in ("x", "both") else []) + ([ax.yaxis] if axis in ("y", "both") else [])
    for a in axes:
        a.set_major_formatter(mt.FuncFormatter(fmt))
        a.set_minor_locator(mt.LogLocator(base=10, subs=minor))
        a.set_minor_formatter(mt.FuncFormatter(fmt))

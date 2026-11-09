"""Matplotlib style, stable per-group colours and figure helpers for telemetry plots."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

# Colour-blind-friendly qualitative palette (Okabe-Ito + Tol muted) for numbered suites,
# indexed by suite number so a suite keeps its colour in every scope and figure.
SUITE_COLORS = [
    "#0072B2",  # 01
    "#D55E00",  # 02
    "#009E73",  # 03
    "#CC79A7",  # 04
    "#E69F00",  # 05
    "#56B4E9",  # 06
    "#882255",  # 07
    "#117733",  # 08
    "#332288",  # 09
    "#AA4499",  # 10
]
PRELIM_COLORS = {
    "Marathon 1M multi-model runs": "#8C564B",
    "Randomized-parameter exploration": "#BCBD22",
    "Detection-criteria sweep (v1/v2)": "#17BECF",
    "Prior response / sensitivity": "#9467BD",
    "Rate-limit benchmark": "#7F7F00",
    "Debug / testing": "#C49C94",
    "Early sizing tests (sim*.sh)": "#393B79",
    "Pre-pipeline runs (mislabelled)": "#B5CF6B",
    "Seeded reference benchmarks": "#DBDB8D",
    "Defunct scripts": "#AD494A",
    "Unclassified legacy runs": "#A6A6A6",
}
OTHER_COLORS = {
    "Pipeline orchestration": "#4D4D4D",
    "Unassigned kn_* jobs": "#999999",
    "Interactive / ad-hoc sessions": "#B0B0B0",
    "Telemetry infrastructure": "#7A7A7A",
    "Unclassified": "#C8C8C8",
}
_FALLBACK = ["#393B79", "#637939", "#8C6D31", "#843C39", "#7B4173", "#3182BD", "#E6550D", "#31A354"]

STATE_COLORS = {
    "COMPLETED": "#009E73",
    "OUT_OF_MEMORY": "#D55E00",
    "FAILED": "#CC79A7",
    "TIMEOUT": "#E69F00",
    "CANCELLED": "#999999",
    "NODE_FAIL": "#882255",
    "RUNNING": "#56B4E9",
    "PENDING": "#0072B2",
}


def apply_style() -> None:
    """Readable, print-friendly defaults."""
    plt.style.use("default")
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "figure.constrained_layout.use": True,
    })


def group_color(group: str) -> str:
    """Stable colour for a classification group."""
    m = re.match(r"^Suite (\d+)", group or "")
    if m:
        return SUITE_COLORS[(int(m.group(1)) - 1) % len(SUITE_COLORS)]
    if group in PRELIM_COLORS:
        return PRELIM_COLORS[group]
    if group in OTHER_COLORS:
        return OTHER_COLORS[group]
    return _FALLBACK[sum(map(ord, group or "")) % len(_FALLBACK)]


def group_palette(groups: Iterable[str]) -> Dict[str, str]:
    return {g: group_color(g) for g in groups}


def short_group(group: str, width: int = 28) -> str:
    """Compact tick label: ``"Suite 02: Granular Criteria Sweep"`` -> ``"S02 Granular Criteria Sweep"``."""
    m = re.match(r"^Suite (\d+):\s*(.*)$", group or "")
    text = f"S{m.group(1)} {m.group(2)}" if m else (group or "")
    return text if len(text) <= width else text[: width - 1] + "…"


def short_name(name: str, width: int = 36) -> str:
    """Middle-truncate a job name (keeps the distinguishing suffix), dropping ``.sh``."""
    name = re.sub(r"\.sh$", "", name or "")
    if len(name) <= width:
        return name
    head = (width - 1) // 2
    tail = width - 1 - head
    return name[:head] + "\u2026" + name[-tail:]


def legend_outside(ax, loc: str = "upper left", ncol: int = 1, title: Optional[str] = None, **kw):
    """Place the legend to the right of the axes so it never covers data."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(handles, labels, loc=loc, bbox_to_anchor=(1.01, 1.0), ncol=ncol,
                     title=title, borderaxespad=0.0, **kw)


def legend_below(fig, handles, labels, ncol: int = 4, title: Optional[str] = None):
    """Figure-level legend below all axes."""
    if not handles:
        return None
    return fig.legend(handles, labels, loc="outside lower center", ncol=ncol, title=title)


def save_figure(fig, name: str, fig_dir: Optional[str]) -> Optional[Path]:
    """Save ``fig`` as ``<fig_dir>/<name>.png`` when ``fig_dir`` is set."""
    if not fig_dir:
        return None
    out = Path(fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    return path


__all__ = ["apply_style", "group_color", "group_palette", "short_group", "short_name", "legend_outside",
           "legend_below", "save_figure", "STATE_COLORS"]

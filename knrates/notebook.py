"""One-call setup for the analysis notebooks.

Typical first cell::

    import sys; from pathlib import Path
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "knrates").is_dir())
    sys.path.insert(0, str(ROOT))
    for m in [m for m in list(sys.modules) if m.split(".")[0] == "knrates"]:
        del sys.modules[m]          # always use the repository copy (a kernel may predate a git pull)
    from knrates import notebook as NB, figures as F, analysis as A, plotting as P, literature as L
    ctx = NB.setup("latest")

``ctx`` carries the snapshot and the Suite 01 (+ optional Suite 07) tables that most sections need, plus
``ctx.save(fig, name)``, which writes PDF+PNG only when ``save_figures`` is on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import analysis as A
from . import plotting as P
from .snapshot import REPO_ROOT, Snapshot, load_snapshot


@dataclass
class Context:
    snap: Snapshot
    runs: pd.DataFrame          # per-realisation results (Suite 01, plus Suite 07 if requested)
    tm: pd.DataFrame            # recovered injections per explosion month
    zh: pd.DataFrame            # recovered injections per redshift bin
    src: pd.DataFrame           # recovered sources (inclination, distance, light-curve sampling)
    lim: pd.DataFrame           # pooled limits, one row per model (indexed by model)
    models: List[str]
    save_figures: bool = False
    fig_dir: Path = field(default_factory=lambda: REPO_ROOT / "figures")

    def save(self, fig, name: str) -> None:
        if self.save_figures:
            for p in P.savefig(fig, name, self.fig_dir):
                print("saved", Path(p).relative_to(REPO_ROOT))


def _cat(snap: Snapshot, name: str, with07: bool) -> pd.DataFrame:
    parts = [snap.get(f"suite01_{name}", pd.DataFrame())]
    if with07:
        parts.append(snap.get(f"suite07_{name}", pd.DataFrame()))
    return pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else parts[0]


def setup(snapshot: str = "latest", *, style: str = "notebook", labels: str = "working",
          include_metzger: bool = False, include_suite07: bool = False, save_figures: bool = False,
          fig_dir: Optional[str] = None, quiet: bool = False) -> Context:
    """Load a snapshot and apply the standard model selection.

    By default only the Suite 01 POSSIS models are kept: the one-zone Metzger models are dropped
    (``include_metzger``) and the Suite 07 GW170817 parameter set, which differs from Suite 01 only by 0.1 dex in
    the dynamical ejecta mass, is dropped (``include_suite07``).
    """
    P.use_style(style)
    P.set_label_mode(labels)
    P.DISAMBIGUATE = include_suite07   # with one parameter set the labels need no suite / M_dyn tag
    pd.set_option("display.width", 160, "display.max_columns", 40, "display.float_format", "{:.4g}".format)
    snap = load_snapshot(snapshot)
    if not include_metzger:
        snap = snap.exclude_models(lambda m: "Metzger" in m)
    if not include_suite07:
        snap = snap.exclude_models(lambda m: m.startswith("GW170817_"))
    runs = _cat(snap, "runs", include_suite07)
    models = [m for m in P.MODEL_STYLE if m in set(runs.get("model", []))]
    ctx = Context(snap=snap, runs=runs, tm=_cat(snap, "texp_month", include_suite07),
                  zh=_cat(snap, "z_hist", include_suite07), src=_cat(snap, "sources", include_suite07),
                  lim=A.headline_limits(runs, models).set_index("model"), models=models,
                  save_figures=save_figures, fig_dir=REPO_ROOT / (fig_dir or "figures"))
    if not quiet:
        print(snap.label)
        print("models:", ", ".join(models))
    return ctx


def md(text: str) -> None:
    """Render a Markdown string in the notebook (prints it elsewhere)."""
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except ImportError:  # pragma: no cover
        print(text)


def finding(text: str) -> None:
    """Render one computed result line, visually set apart from the code output."""
    md("> **Result.** " + text)


def all_limits(snapshot: str = "latest") -> pd.DataFrame:
    """Pooled limits of every fixed-parameter model (Suites 01 and 07, Metzger included), indexed by model.

    Unlike :func:`setup` this leaves the plotting style and labels untouched.
    """
    snap = load_snapshot(snapshot)
    runs = pd.concat([d for d in (snap.get("suite01_runs"), snap.get("suite07_runs")) if d is not None and len(d)],
                     ignore_index=True)
    return A.headline_limits(runs).set_index("model")

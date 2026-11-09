"""Load compiled results snapshots (see :mod:`knrates.collect`).

Typical notebook usage::

    from knrates import load_snapshot, list_snapshots
    SNAPSHOT = "latest"          # or "20261109T183000Z", a path, or "live"
    snap = load_snapshot(SNAPSHOT)
    runs01 = snap["suite01_runs"]
    print(snap.describe())

``which`` may be

* ``None`` / ``"latest"`` -- ``snapshots/knrates_results_latest.zip``
* a time stamp (or unique prefix of one), e.g. ``"20261109"`` -- the matching
  ``snapshots/knrates_results_<stamp>.zip`` (``"newest"`` picks the most recent)
* a path to any snapshot ZIP
* ``"live"`` -- build in memory from ``results/`` without writing anything
  (requires the raw results; slow).
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO_ROOT / "snapshots"
LATEST_NAME = "knrates_results_latest.zip"


@dataclass
class Snapshot:
    tables: Dict[str, pd.DataFrame]
    manifest: dict
    path: Optional[Path] = None

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"table '{name}' not in snapshot; available: {sorted(self.tables)}")
        return self.tables[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tables

    def get(self, name: str, default=None):
        return self.tables.get(name, default)

    @property
    def label(self) -> str:
        src = self.path.name if self.path else "live build"
        return f"{src} (created {self.manifest.get('created_utc', '?')}, git {self.manifest.get('git', {}).get('commit', '')[:8]})"

    def describe(self) -> str:
        lines = [f"Snapshot: {self.label}"]
        for name, df in sorted(self.tables.items()):
            lines.append(f"  {name:<28s} {len(df):>9,d} rows")
        w = self.manifest.get("warnings", [])
        if w:
            lines.append(f"  {len(w)} build warning(s):")
            lines.extend(f"    - {x}" for x in w)
        sk = self.manifest.get("skipped_files", [])
        if sk:
            lines.append(f"  {len(sk)} skipped (unreadable) result file(s)")
        return "\n".join(lines)


def list_snapshots(directory: Optional[Union[str, Path]] = None) -> List[Path]:
    """All snapshot archives, time-stamped ones sorted oldest -> newest, 'latest' last."""
    d = Path(directory) if directory is not None else SNAPSHOT_DIR
    stamped = sorted(p for p in d.glob("knrates_results_*.zip") if p.name != LATEST_NAME)
    latest = [d / LATEST_NAME] if (d / LATEST_NAME).exists() else []
    return stamped + latest


def _resolve(which: Optional[Union[str, Path]]) -> Path:
    if which is None or str(which) == "latest":
        p = SNAPSHOT_DIR / LATEST_NAME
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Build it with `python -m knrates.collect` from the repo root.")
        return p
    p = Path(which)
    if p.suffix == ".zip" and p.exists():
        return p
    stamped = [x for x in list_snapshots() if x.name != LATEST_NAME]
    if str(which) == "newest":
        if not stamped:
            raise FileNotFoundError("no time-stamped snapshots found")
        return stamped[-1]
    matches = [x for x in stamped if x.stem.replace("knrates_results_", "").startswith(str(which))]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no snapshot matches '{which}'. Available: {[x.name for x in list_snapshots()]}")
    raise ValueError(f"'{which}' is ambiguous: {[x.name for x in matches]}")


def load_snapshot(which: Optional[Union[str, Path]] = "latest") -> Snapshot:
    """Load a snapshot archive (or build one live when ``which == 'live'``)."""
    if str(which) == "live":
        from .collect import build

        tables, manifest, errors = build()
        if errors:
            raise RuntimeError("live build failed validation:\n" + "\n".join(errors))
        return Snapshot(tables, manifest, None)
    path = _resolve(which)
    tables: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        for name in zf.namelist():
            if name.endswith(".csv"):
                data = zf.read(name)
                tables[name[:-4]] = pd.read_csv(io.BytesIO(data), low_memory=False) if data.strip() else pd.DataFrame()
    for name in ("suite01_runs", "suite07_runs"):
        if name in tables and "task_id" in tables[name]:
            tables[name]["task_id"] = tables[name]["task_id"].astype(str)
    return Snapshot(tables, manifest, path)

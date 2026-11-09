"""Optional, failure-tolerant capture of Slurm FairShare snapshots via ``sshare``.

When the harvester runs on the cluster (e.g. from ``scripts/cron_update_telemetry.sh``),
:func:`capture_sshare_snapshot` runs::

    sshare -u $USER -l -P

and appends one row per association to ``telemetry/fairshare_snapshots.csv`` with a
``snapshot_time`` column.  Over time this gives the *real* FairShare factor history
alongside the registry.  Nothing is ever fabricated: if ``sshare`` is missing (e.g. on a
laptop), times out, or prints something unparseable, the function returns ``None`` and
writes nothing.

Pure Python (no pandas) so it is safe to import in the Slurm-side cron environment.
"""

from __future__ import annotations

import csv
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

SNAPSHOT_RELPATH = Path("telemetry") / "fairshare_snapshots.csv"

#: Output columns and the ``sshare`` header names they are read from (case-insensitive).
SSHARE_FIELDS: Dict[str, tuple] = {
    "account": ("account",),
    "user": ("user",),
    "raw_shares": ("rawshares",),
    "norm_shares": ("normshares",),
    "raw_usage": ("rawusage",),
    "norm_usage": ("normusage",),
    "effective_usage": ("effectvusage", "effectiveusage"),
    "fairshare": ("fairshare",),
    "level_fs": ("levelfs",),
    "grp_tres_raw": ("grptresraw",),
}
SNAPSHOT_COLUMNS = ["snapshot_time"] + list(SSHARE_FIELDS)


def parse_sshare_output(text: str, snapshot_time: Optional[str] = None) -> List[Dict[str, str]]:
    """Parse ``sshare -l -P`` (pipe-delimited, header first) into snapshot rows.

    Rows without a FairShare value (e.g. parent account lines) are kept, so the account
    tree context is preserved; blank cells stay blank.
    """
    if not text or not text.strip():
        return []
    lines = [ln.rstrip("\n") for ln in text.strip().splitlines() if ln.strip()]
    header_idx = next((i for i, ln in enumerate(lines) if "|" in ln and "account" in ln.lower()), None)
    if header_idx is None:
        return []
    header = [h.strip().lower().replace(" ", "") for h in lines[header_idx].split("|")]
    stamp = snapshot_time or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    rows: List[Dict[str, str]] = []
    for ln in lines[header_idx + 1:]:
        if set(ln.strip()) <= {"-", "|", " "}:
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 2:
            continue
        rec = {"snapshot_time": stamp}
        for out_col, names in SSHARE_FIELDS.items():
            val = ""
            for name in names:
                if name in header:
                    idx = header.index(name)
                    val = parts[idx] if idx < len(parts) else ""
                    break
            rec[out_col] = val
        rows.append(rec)
    return rows


def capture_sshare_snapshot(
    project_root: Optional[Path] = None,
    user: Optional[str] = None,
    out_path: Optional[Path] = None,
    timeout: float = 30.0,
) -> Optional[List[Dict[str, str]]]:
    """Run ``sshare`` and append the parsed rows to the snapshot CSV.

    Returns the rows written, or ``None`` when ``sshare`` is unavailable or fails (never
    raises).
    """
    exe = shutil.which("sshare")
    if not exe:
        return None
    user = user or os.environ.get("USER") or os.environ.get("LOGNAME")
    cmd = [exe, "-l", "-P"]
    if user:
        cmd[1:1] = ["-u", user]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        rows = parse_sshare_output(res.stdout)
    except Exception as exc:  # pragma: no cover - depends on cluster state
        sys.stderr.write(f"Warning: sshare snapshot skipped: {exc}\n")
        return None
    if not rows:
        return None
    path = Path(out_path) if out_path else Path(project_root or Path.cwd()) / SNAPSHOT_RELPATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLUMNS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"Warning: could not write fairshare snapshot: {exc}\n")
        return None
    return rows


def load_fairshare_snapshots(path: Path) -> List[Dict[str, str]]:
    """Read previously captured snapshots (empty list when the file does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


__all__ = ["SNAPSHOT_RELPATH", "parse_sshare_output", "capture_sshare_snapshot", "load_fairshare_snapshots"]

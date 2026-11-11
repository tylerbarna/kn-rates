#!/usr/bin/env python3
"""DEPRECATED generator -- ``notebooks/telemetry_job_resources.ipynb`` is now the source of truth.

The telemetry notebook used to be generated from code in this file, which made the two drift
apart (hard-coded numbers in markdown, plots that no longer matched the data).  Edit the
notebook directly; the analysis logic lives in the ``telemetry`` package
(``telemetry.classify``, ``telemetry.frame``, ``telemetry.accounting``, ``telemetry.plotting``).

This wrapper is kept so existing workflows do not break.  It can:

* ``--check``   validate the notebook (nbformat-4.5 cell ids, settings cell first, no stale outputs
                required) -- exit status 1 on problems;
* ``--execute`` run the notebook top-to-bottom in place with ``jupyter nbconvert``
                (from the ``notebooks/`` directory; the notebook locates the repository itself).

Usage::

    python scripts/build_telemetry_notebook.py --check
    python scripts/build_telemetry_notebook.py --execute [--scope preliminary]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "notebooks" / "telemetry_job_resources.ipynb"
SCOPES = ("numbered", "preliminary", "all")


def check_notebook(path: Path = NOTEBOOK) -> list:
    """Return a list of problems found in the notebook (empty list = OK)."""
    problems = []
    if not path.exists():
        return [f"{path} does not exist"]
    nb = json.loads(path.read_text(encoding="utf-8"))
    ids = []
    for i, cell in enumerate(nb.get("cells", [])):
        cid = cell.get("id")
        if not isinstance(cid, str) or not cid:
            problems.append(f"cell {i} has no valid id")
        ids.append(cid)
    if len(set(ids)) != len(ids):
        problems.append("duplicate cell ids")
    code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not code or "ANALYSIS_SCOPE" not in "".join(code[0].get("source", [])):
        problems.append("first code cell must be the settings cell defining ANALYSIS_SCOPE")
    return problems


def execute_notebook(path: Path = NOTEBOOK, scope: str | None = None, timeout: int = 900) -> int:
    """Execute the notebook in place (optionally overriding ANALYSIS_SCOPE for this run only)."""
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))
    target = path
    if scope:
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code" and "ANALYSIS_SCOPE" in "".join(cell["source"]):
                cell["source"] = [line if not line.startswith("ANALYSIS_SCOPE") else f'ANALYSIS_SCOPE = "{scope}"\n'
                                  for line in cell["source"]]
                break
        target = path.with_name(f"{path.stem}_{scope}.ipynb")
        target.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Executing a copy with ANALYSIS_SCOPE={scope!r}: {target}")
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           f"--ExecutePreprocessor.timeout={timeout}", target.name]
    return subprocess.call(cmd, cwd=target.parent, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="validate the notebook structure")
    parser.add_argument("--execute", action="store_true", help="execute the notebook in place")
    parser.add_argument("--scope", choices=SCOPES, help="with --execute: run a copy with this ANALYSIS_SCOPE")
    args = parser.parse_args()

    if not (args.check or args.execute):
        print("scripts/build_telemetry_notebook.py is deprecated: the notebook "
              f"{NOTEBOOK.relative_to(REPO_ROOT)} is the source of truth and is no longer generated.\n"
              "Use --check to validate it or --execute to re-run it.")
        return 0
    status = 0
    if args.check:
        problems = check_notebook()
        for p in problems:
            print(f"PROBLEM: {p}")
        print("notebook OK" if not problems else f"{len(problems)} problem(s)")
        status |= 1 if problems else 0
    if args.execute:
        status |= execute_notebook(scope=args.scope)
    return status


if __name__ == "__main__":
    sys.exit(main())

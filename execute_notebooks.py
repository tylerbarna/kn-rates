#!/usr/bin/env python3
"""Execute project notebooks in place with a real Jupyter kernel (nbclient).

Usage::

    python execute_notebooks.py                       # all notebooks in notebooks/
    python execute_notebooks.py notebooks/01_*.ipynb  # selected notebooks
    python execute_notebooks.py --timeout 1800 --keep-going

Notebooks are executed with ``notebooks/`` as the working directory (they locate the
repository root and the results snapshot themselves).  A notebook that raises is
reported and, unless ``--keep-going`` is given, stops the run with exit status 1;
the failing notebook is still written so the traceback can be inspected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent


def execute_notebook(nb_path: Path, timeout: int = 1800, kernel: str = "python3") -> bool:
    nb_path = Path(nb_path)
    print(f"Executing {nb_path} ...", flush=True)
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name=kernel,
                            resources={"metadata": {"path": str(nb_path.resolve().parent)}})
    ok = True
    try:
        client.execute()
    except CellExecutionError as exc:
        ok = False
        print(f"  FAILED: {str(exc).splitlines()[-1] if str(exc) else exc}", flush=True)
    nbformat.write(nb, nb_path)
    print(f"  {'ok' if ok else 'written with errors'}: {nb_path}", flush=True)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("notebooks", nargs="*", help="notebook paths (default: notebooks/*.ipynb)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failing notebook")
    args = ap.parse_args(argv)
    targets = [Path(p) for p in args.notebooks] or sorted((ROOT / "notebooks").glob("*.ipynb"))
    failed = []
    for nb in targets:
        if not execute_notebook(nb, args.timeout):
            failed.append(nb)
            if not args.keep_going:
                break
    if failed:
        print("Failed notebooks:", *map(str, failed), sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

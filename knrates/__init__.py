"""Shared analysis utilities for the ZTF 8-year kilonova rate-limit project.

Sub-modules
-----------
``knrates.stats``       Poisson upper limits and pooled Monte Carlo estimators.
``knrates.collect``     Builds the version-controlled results snapshot from ``results/``
                        (run as ``python -m knrates.collect``).
``knrates.snapshot``    Loads a snapshot (latest, a time-stamped one, or live from ``results/``).
``knrates.literature``  Published reference values used for comparisons (with citations).
``knrates.plotting``    Publication plotting style, model labels and colours.
"""

from .snapshot import Snapshot, list_snapshots, load_snapshot  # noqa: F401

__all__ = ["Snapshot", "list_snapshots", "load_snapshot"]

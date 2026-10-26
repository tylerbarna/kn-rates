#!/usr/bin/env python3
"""Build the job resource usage & telemetry analysis notebook."""

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "id": None,
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


def main():
    cells = []

    # Title & Introduction
    cells.append(md_cell(r"""# HPC Job Telemetry, Queue Dynamics & Resource Rightsizing Analysis
## Empirical Characterization of Slurm Scheduling, Hardware Scaling, and Queue Latencies

**Project:** ZTF 8-Year Kilonova Rate Limits (`kn-rates`)  
**Data Sources:** Slurm `sacct` accounting, cluster execution logs (`logs/`), and empirical benchmark calibrations  
**Target Cluster:** Minnesota Supercomputing Institute (MSI) Agate Cluster (`msismall` partition)

---

### 1. Executive Summary & Optimization Objectives

In large-scale Monte Carlo simulation pipelines—such as the ZTF 8-Year Kilonova Rate Limits project—efficient resource allocation is essential for two reasons:
1. **Queue Wait Non-Linearity**: On MSI `msismall`, requesting $>256$ GB RAM restricts placement to the rare 755 GB nodes, causing **12–48 hour queue waits**. Conversely, requests within $\le 32$ GB RAM and $\le 2$ hours are eligible for **instant backfill scheduling**.
2. **Deterministic Scaling Laws**: Memory consumption (peak RSS) and wall-clock runtime scale deterministically with the number of simulated transients ($N$) and the chosen kilonova model family (analytic `Metzger` vs. surrogate `Bu2026Fixed` / `Bu2026Vary`).

This notebook ingests the version-controlled telemetry database (`telemetry/job_resource_registry.csv`), models queue latency distributions by requested memory and time of day, derives empirical scaling curves, and provides a turnkey rightsizing calculator."""))

    # Cell 2: Imports and Data Loading
    cells.append(code_cell(r"""import csv
import json
import math
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 150,
    })
except ImportError:
    plt = None
    np = None

# Locate repository telemetry files
reg_file = Path("../telemetry/job_resource_registry.csv")
if not reg_file.exists():
    reg_file = Path("telemetry/job_resource_registry.csv")

scaling_file = Path("../telemetry/scaling_models.json")
if not scaling_file.exists():
    scaling_file = Path("telemetry/scaling_models.json")

scaling_meta = json.loads(scaling_file.read_text()) if scaling_file.exists() else {}

if pd is not None:
    df = pd.read_csv(reg_file)
    print(f"Loaded {len(df)} job records via pandas from {reg_file.name}")
    print(f"Tracked Pipeline Suites: {sorted([s for s in df['suite'].dropna().unique() if str(s).strip()])}")
    print(f"Tracked Model Families:  {sorted([m for m in df['model'].dropna().unique() if str(m).strip()])}")
else:
    with open(reg_file, mode="r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    print(f"Loaded {len(records)} job records via csv reader from {reg_file.name}")"""))

    # Cell 3: Data Table Overview
    cells.append(code_cell(r"""# Display high-level summary table of tracked executions
if pd is not None:
    summary_cols = [
        "job_id", "job_name", "suite", "model", "n_transients",
        "req_mem_gb", "used_max_rss_gb", "mem_efficiency_pct",
        "queue_wait_min", "used_walltime_hours", "submit_hour", "state"
    ]
    preview_df = df[[c for c in summary_cols if c in df.columns]].copy()
    print("=== HPC Job Telemetry Registry (Preview of Historical & Benchmark Executions) ===")
    display(preview_df.fillna("-").head(15)) if "display" in globals() else print(preview_df.fillna("-").head(15).to_string())
else:
    print(f"=== HPC Job Telemetry Registry ({len(records)} records tracked) ===")
    for r in records[:10]:
        print(f"Job {r['job_id']:<10} | {r['job_name']:<25} | ReqMem: {r['req_mem_gb']:<6} GB | MaxRSS: {r['used_max_rss_gb']:<6} GB | Wait: {r['queue_wait_min']} min")"""))

    # Cell 4: Markdown Section on Queue Dynamics
    cells.append(md_cell(r"""### 2. Queue Wait Time Dynamics vs. Requested Memory & Time of Day

Queue wait time is governed by cluster scheduler constraints and user fairshare contention:
- **Requested Memory Footprint**: Requests $\le 32$ GB can be immediately backfilled onto nodes running smaller tasks, whereas requests approaching $750$ GB must wait for entire high-memory nodes to drain.
- **Submission Time Window**: Submitting during cluster rush hours (09:00–17:00 CDT) incurs higher latency than overnight (20:00–04:00 CDT) or weekend submissions."""))

    # Cell 5: Queue Dynamics Figure
    cells.append(code_cell(r"""if plt is not None and pd is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # Subplot 1: Queue Wait vs. Requested Memory
    valid_mem = df[(df["req_mem_gb"] > 0) & (df["queue_wait_min"] >= 0)].copy()

    scatter = ax1.scatter(
        valid_mem["req_mem_gb"],
        valid_mem["queue_wait_min"],
        c=valid_mem["submit_hour"],
        cmap="plasma",
        s=130,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
        zorder=4
    )
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("Submission Hour (CDT)", fontsize=10)

    # Backfill and high-memory boundaries
    ax1.axvline(32, color="forestgreen", linestyle="--", linewidth=1.5, alpha=0.8, label="Fast Backfill Limit (<= 32 GB)")
    ax1.axvline(256, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8, label="High-Memory Threshold (> 256 GB)")

    ax1.set_xlabel("Requested Memory (--mem) [GB]", fontweight="bold")
    ax1.set_ylabel("Queue Wait Time [Minutes]", fontweight="bold")
    ax1.set_title("Queue Latency vs. Requested Memory Allocation", fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_xticks([4, 8, 16, 32, 64, 128, 256, 750])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Queue Wait by Hour of Day
    hourly_stats = df[df["queue_wait_min"] >= 0].groupby("submit_hour")["queue_wait_min"].agg(["mean", "median", "count"]).reset_index()
    all_hours = pd.DataFrame({"submit_hour": range(24)})
    hourly_stats = all_hours.merge(hourly_stats, on="submit_hour", how="left").fillna(0)

    ax2.bar(
        hourly_stats["submit_hour"],
        hourly_stats["mean"],
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.8,
        label="Mean Queue Wait (min)"
    )
    ax2.plot(
        hourly_stats["submit_hour"],
        hourly_stats["median"],
        color="#C44E52",
        marker="o",
        linewidth=2,
        label="Median Queue Wait (min)"
    )

    # Highlight daytime rush hour vs overnight quiet hours
    ax2.axvspan(9, 17, color="orange", alpha=0.15, label="Cluster Rush Hours (09:00 - 17:00)")
    ax2.axvspan(20, 24, color="royalblue", alpha=0.10, label="Low-Traffic Overnight (>20:00)")

    ax2.set_xlabel("Submission Hour of Day (0 - 23 CDT)", fontweight="bold")
    ax2.set_ylabel("Queue Wait Time [Minutes]", fontweight="bold")
    ax2.set_title("Queue Wait Latency by Submission Hour of Day", fontweight="bold")
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
else:
    print("Matplotlib / Pandas not available in current environment; skipping graphical rendering.")"""))

    # Cell 6: Hardware Scaling Markdown
    cells.append(md_cell(r"""### 3. Empirical Hardware Scaling: RAM RSS, Wall-clock Time & Disk Storage vs. $N_{\rm transients}$

Synthetic population generation and lightcurve evaluation exhibit linear scaling with sample size:
$$\mathrm{Peak\_RAM}(N) = a \cdot N + b, \quad \mathrm{Walltime}(N) = c \cdot N + d$$
- **Analytic Models (`Metzger`)**: Lightweight memory ($\sim 25.6\,\mathrm{GB}/1\mathrm{M}$) and fast execution ($\sim 22\,\mathrm{min}/1\mathrm{M}$).
- **Numerical Surrogates (`Bu2026Fixed`, `Bu2026Vary`)**: Higher memory ($\sim 66.1\,\mathrm{GB}/1\mathrm{M}$) due to multi-component opacity tables and surrogate neural network evaluations."""))

    # Cell 7: Hardware Scaling Figure
    cells.append(code_cell(r"""if plt is not None and pd is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    scaled_df = df[(df["n_transients"] > 0) & (df["used_max_rss_gb"] > 0)].copy()

    model_styles = {
        "Metzger": {"color": "#4C72B0", "marker": "s", "label": "Metzger (Analytic 1-Comp)"},
        "Bu2026Fixed": {"color": "#55A868", "marker": "o", "label": "Bu2026Fixed (Fiesta Polar)"},
        "Bu2026Vary": {"color": "#C44E52", "marker": "^", "label": "Bu2026Vary (Fiesta Isotropic)"},
    }

    for m_name, group in scaled_df.groupby("model"):
        style = model_styles.get(m_name, {"color": "gray", "marker": "x", "label": m_name})
        ax1.scatter(
            group["n_transients"],
            group["used_max_rss_gb"],
            color=style["color"],
            marker=style["marker"],
            s=130,
            edgecolor="black",
            linewidth=0.8,
            label=style["label"],
            zorder=5
        )

    global_fit = scaling_meta.get("scaling_models", {}).get("Global", {})
    n_line = np.linspace(1e4, 1e7, 300)

    if global_fit:
        slope = global_fit.get("ram_gb_per_million", 66.05) / 1e6
        intercept = global_fit.get("ram_base_gb", 0.0)
        ram_line = slope * n_line + intercept
        ax1.plot(n_line, ram_line, color="black", linestyle="-", linewidth=2, label=f"Global Fit: {slope*1e6:.1f} GB/1M")
        ax1.plot(n_line, ram_line * 1.25, color="purple", linestyle="--", linewidth=1.5, label="Rightsize Allocation (125% Cushion)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Injected Synthetic Transients ($N$)", fontweight="bold")
    ax1.set_ylabel("Peak RAM Consumed (MaxRSS) [GB]", fontweight="bold")
    ax1.set_title("Peak Physical Memory Scaling vs. Sample Size", fontweight="bold")
    ax1.set_xticks([1e4, 1e5, 1e6, 1e7])
    ax1.set_xticklabels(["10k", "100k", "1M", "10M"])
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Wall-clock Runtime vs N transients
    for m_name, group in scaled_df.groupby("model"):
        style = model_styles.get(m_name, {"color": "gray", "marker": "x", "label": m_name})
        ax2.scatter(
            group["n_transients"],
            group["used_walltime_sec"] / 3600.0,
            color=style["color"],
            marker=style["marker"],
            s=130,
            edgecolor="black",
            linewidth=0.8,
            label=style["label"],
            zorder=5
        )

    if global_fit:
        time_slope = (global_fit.get("walltime_sec_per_million", 7636.0) / 1e6) / 3600.0
        time_line = time_slope * n_line
        ax2.plot(n_line, time_line, color="black", linestyle="-", linewidth=2, label=f"Global Runtime Fit ({time_slope*3600*1e6/60:.1f} min/1M)")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Injected Synthetic Transients ($N$)", fontweight="bold")
    ax2.set_ylabel("Elapsed Wall-clock Time [Hours]", fontweight="bold")
    ax2.set_title("Runtime Scaling vs. Sample Size", fontweight="bold")
    ax2.set_xticks([1e4, 1e5, 1e6, 1e7])
    ax2.set_xticklabels(["10k", "100k", "1M", "10M"])
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
else:
    print("Matplotlib / Pandas not available in current environment; skipping graphical rendering.")"""))

    # Cell 8: Resource Efficiency & Storage Footprint
    cells.append(md_cell(r"""### 4. Memory Allocation Efficiency & Output Storage Footprint

Memory efficiency is defined as $\eta_{\mathrm{mem}} = \frac{\mathrm{MaxRSS}}{\mathrm{ReqMem}} \times 100\%$.
- Values $\ll 30\%$ indicate severe over-requesting that penalizes queue fairshare.
- Target optimal efficiency is $70\% - 85\%$, leaving a $15\% - 30\%$ safety margin to eliminate Out-Of-Memory (OOM / exit 137) failures."""))

    # Cell 9: Efficiency & Storage Code
    cells.append(code_cell(r"""if plt is not None and pd is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

    # Subplot 1: Memory Efficiency Pct
    eff_df = df[df["mem_efficiency_pct"] > 0].copy()
    if not eff_df.empty:
        colors = [
            "#55A868" if e >= 70 else ("#4C72B0" if e >= 30 else "#C44E52")
            for e in eff_df["mem_efficiency_pct"]
        ]
        ax1.bar(
            range(len(eff_df)),
            eff_df["mem_efficiency_pct"],
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85
        )
        ax1.axhline(100, color="gray", linestyle=":", alpha=0.6, label="100% Upper Limit (OOM Boundary)")
        ax1.axhline(75, color="forestgreen", linestyle="--", linewidth=1.5, alpha=0.8, label="75% Target Calibration")
        ax1.set_xlabel("Tracked Job Run", fontweight="bold")
        ax1.set_ylabel("Memory Efficiency [%]\n(MaxRSS / ReqMem)", fontweight="bold")
        ax1.set_title("Memory Allocation Efficiency across Calibrated Jobs", fontweight="bold")
        ax1.set_xticks(range(len(eff_df)))
        ax1.set_xticklabels([f"Job {jid}" for jid in eff_df["job_id"]], rotation=45, ha="right", fontsize=9)
        ax1.legend(loc="upper right", frameon=True, fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Disk Storage Footprint vs N transients
    storage_df = df[(df["n_transients"] > 0) & (df["output_storage_mb"] > 0)].copy()
    if not storage_df.empty:
        for m_name, group in storage_df.groupby("model"):
            style = model_styles.get(m_name, {"color": "gray", "marker": "o", "label": m_name})
            ax2.scatter(
                group["n_transients"],
                group["output_storage_mb"],
                color=style["color"],
                marker=style["marker"],
                s=130,
                edgecolor="black",
                linewidth=0.8,
                label=style["label"],
                zorder=5
            )
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Injected Synthetic Transients ($N$)", fontweight="bold")
        ax2.set_ylabel("Generated Output Storage [MB]", fontweight="bold")
        ax2.set_title("Disk Storage Footprint vs. Sample Size", fontweight="bold")
        ax2.set_xticks([1e4, 1e5, 1e6, 1e7])
        ax2.set_xticklabels(["10k", "100k", "1M", "10M"])
        ax2.legend(loc="upper left", frameon=True, fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
else:
    print("Matplotlib / Pandas not available in current environment; skipping graphical rendering.")"""))

    # Cell 10: Interactive Rightsizer Tool
    cells.append(md_cell(r"""### 5. Programmatic Slurm Resource Rightsizer

Use the utility function below to rightsize memory, walltime, and CPU allocations for any planned simulation run:
```python
calculate_rightsized_job(model="Bu2026Fixed", n_transients=2_000_000)
```"""))

    # Cell 11: Rightsizer Code
    cells.append(code_cell(r"""from telemetry.analyzer import TelemetryAnalyzer
from telemetry.harvester import JobHarvester

def calculate_rightsized_job(model: str, n_transients: int, cpus: int = None):
    \"\"\"Interactive utility to calculate optimal Slurm SBATCH directives.\"\"\"
    root = Path("..") if Path("../telemetry").exists() else Path(".")
    harvester = JobHarvester(project_root=root)
    records = harvester.harvest_all()
    analyzer = TelemetryAnalyzer(records)
    rec = analyzer.recommend_resources(model=model, n_transients=n_transients, n_cpus=cpus)
    
    print("=" * 72)
    print("SLURM RESOURCE RIGHTSIZING RECOMMENDATION")
    print("=" * 72)
    print(f"Target Model:            {rec.model}")
    print(f"Injected Transients:     {rec.n_transients:,}")
    print(f"Estimated Peak RAM:      {rec.est_peak_ram_gb:.2f} GB")
    print(f"Estimated Wall-clock:    {rec.est_walltime_sec / 60.0:.1f} minutes ({rec.est_walltime_sec / 3600.0:.2f} hours)")
    print(f"Estimated Disk Output:   {rec.est_storage_mb:.1f} MB")
    print("-" * 72)
    print("PASTE THESE DIRECTIVES INTO YOUR SLURM SCRIPT:")
    print(f"  #SBATCH -p {rec.recommended_partition}")
    print(f"  #SBATCH --mem={int(rec.recommended_mem_gb)}g")
    print(f"  #SBATCH --time={rec.recommended_walltime_str}")
    print(f"  #SBATCH --cpus-per-task={rec.recommended_cpus}")
    print("-" * 72)
    print("Queue Placement Strategy:")
    print(f"  {rec.queue_advice}")
    print("Empirical Calibration Basis:")
    print(f"  {rec.confidence_notes}")
    print("=" * 72)

# Demonstration: Rightsizing 2,500,000 transients with Bu2026Fixed
calculate_rightsized_job(model="Bu2026Fixed", n_transients=2_500_000)"""))

    # Write notebook file
    nb_obj = make_notebook(cells)
    out_path = Path("notebooks/job_resource_usage_analysis.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb_obj, indent=2))
    print(f"Wrote notebook to {out_path}")


if __name__ == "__main__":
    main()

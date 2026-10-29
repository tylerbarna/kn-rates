#!/usr/bin/env python3
"""Build the job resource usage & telemetry analysis notebook with enhanced efficiency profiling."""

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
## Empirical Characterization of Slurm Scheduling, Efficiency Metrics (reportseff), and Hardware Scaling

**Project:** ZTF 8-Year Kilonova Rate Limits (`kn-rates`)  
**Data Sources:** Slurm `reportseff` efficiency accounting, `sacct` kernel metrics, execution logs (`logs/`), and empirical benchmark calibrations  
**Target Cluster:** Minnesota Supercomputing Institute (MSI) Agate Cluster (`msismall` partition)

---

### 1. Executive Summary & Optimization Objectives

In large-scale Monte Carlo simulation pipelines—such as the ZTF 8-Year Kilonova Rate Limits project—efficient resource allocation is essential for two reasons:
1. **Queue Wait Non-Linearity**: On MSI `msismall`, requesting $>256$ GB RAM restricts placement to the rare 755 GB nodes, causing **12–48 hour queue waits**. Conversely, requests within $\le 32$ GB RAM and $\le 2$ hours are eligible for **instant backfill scheduling**.
2. **Efficiency Metrics (`reportseff` & `seff`)**: Over-allocating CPUs and walltime drastically harms cluster fairshare priority:
   - **Time Efficiency (`TimeEff`)**: $\mathrm{Elapsed} / \mathrm{Timelimit}$. Over-requesting walltime (e.g. asking for 24h for a 10-minute job) causes massive scheduling penalties. Values $\ge 100\%$ indicate timeouts.
   - **CPU Efficiency (`CPUEff`)**: $\mathrm{TotalCPU} / (\mathrm{Elapsed} \times \mathrm{ReqCPUS})$. Identifies unparallelized multi-core jobs (e.g. requesting 8 cores for single-threaded code yields $\sim 12.5\%$ efficiency).
   - **Memory Efficiency (`MemEff`)**: $\mathrm{MaxRSS} / \mathrm{ReqMem}$. Target optimal calibration is **70%–85%**, leaving a safe cushion against OOM kills while minimizing fairshare penalties.

This notebook ingests the version-controlled telemetry database (`telemetry/job_resource_registry.csv`), visualizes queue latency distributions, derives empirical scaling curves, profiles hardware efficiency distributions, and provides a turnkey rightsizing calculator."""))

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
    print(f"Data Sources:            {sorted([d for d in df['data_source'].dropna().unique() if str(d).strip()])}")
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
        "req_time_min", "used_walltime_hours", "time_efficiency_pct",
        "req_cpus", "cpu_efficiency_pct", "queue_wait_min", "state", "data_source"
    ]
    preview_df = df[[c for c in summary_cols if c in df.columns]].copy()
    print("=== HPC Job Telemetry Registry (Preview of Historical & Benchmark Executions) ===")
    display(preview_df.fillna("-").head(15)) if "display" in globals() else print(preview_df.fillna("-").head(15).to_string())
else:
    print(f"=== HPC Job Telemetry Registry ({len(records)} records tracked) ===")
    for r in records[:10]:
        print(f"Job {r['job_id']:<10} | {r['job_name']:<25} | ReqMem: {r['req_mem_gb']:<6} GB | MaxRSS: {r['used_max_rss_gb']:<6} GB | TimeEff: {r.get('time_efficiency_pct', '0')}% | CPUEff: {r.get('cpu_efficiency_pct', '0')}% | MemEff: {r.get('mem_efficiency_pct', '0')}%")"""))

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
    cells.append(md_cell(r"""### 3. Empirical Hardware Scaling: RAM RSS & Wall-clock Time vs. $N_{\rm transients}$

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

    # Cell 8: Resource Efficiency Section Markdown
    cells.append(md_cell(r"""### 4. Comprehensive Resource Efficiency Profile: Time, CPU & Memory (`reportseff`)

Using the Slurm efficiency metrics documented in the [MSI Job Sizing Guide](https://userdocs.msi.umn.edu/compute/job_sizing.html), we examine three efficiency dimensions across historical jobs:
1. **Time Efficiency (`TimeEff`)**: $\frac{\mathrm{Elapsed}}{\mathrm{Timelimit}} \times 100\%$. Measures walltime utilization. Low values ($<10\%$) indicate over-requested walltime which incurs queue scheduling penalties; values $\ge 100\%$ identify jobs that hit Slurm wallclock timeouts.
2. **CPU Efficiency (`CPUEff`)**: $\frac{\mathrm{TotalCPU}}{\mathrm{Elapsed} \times \mathrm{ReqCPUS}} \times 100\%$. Measures true CPU core utilization. Low values reveal unparallelized routines occupying multi-core reservations.
3. **Memory Efficiency (`MemEff`)**: $\frac{\mathrm{MaxRSS}}{\mathrm{ReqMem}} \times 100\%$. Measures physical memory consumption against allocation. Target calibration is **70%–85%**."""))

    # Cell 9: Efficiency Distributions Figure
    cells.append(code_cell(r"""if plt is not None and pd is not None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.0))

    # Subplot 1: Time Efficiency Distribution
    valid_teff = df[df["time_efficiency_pct"] > 0]["time_efficiency_pct"]
    if not valid_teff.empty:
        ax1.hist(
            valid_teff,
            bins=np.linspace(0, 150, 31),
            color="#4C72B0",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.85
        )
        ax1.axvline(100, color="crimson", linestyle="--", linewidth=1.8, label="Timeout Threshold (100%)")
        ax1.axvline(valid_teff.median(), color="orange", linestyle="-", linewidth=2, label=f"Median ({valid_teff.median():.1f}%)")
        ax1.axvspan(70, 90, color="forestgreen", alpha=0.15, label="Target Zone (70-90%)")
        ax1.set_xlabel("Time Efficiency [%]\n(Elapsed / Timelimit)", fontweight="bold")
        ax1.set_ylabel("Number of Jobs", fontweight="bold")
        ax1.set_title(f"Time Efficiency Distribution (N={len(valid_teff)})", fontweight="bold")
        ax1.legend(loc="upper right", frameon=True, fontsize=8.5)
        ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: CPU Efficiency Distribution
    # Note: reportseff records contain true kernel-level CPU efficiency
    rep_df = df[df["data_source"] == "reportseff"].copy()
    valid_cpueff = rep_df[rep_df["cpu_efficiency_pct"] > 0]["cpu_efficiency_pct"]
    if not valid_cpueff.empty:
        ax2.hist(
            valid_cpueff,
            bins=np.linspace(0, 100, 21),
            color="#55A868",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.85
        )
        ax2.axvline(valid_cpueff.median(), color="orange", linestyle="-", linewidth=2, label=f"Median ({valid_cpueff.median():.1f}%)")
        ax2.axvspan(70, 100, color="forestgreen", alpha=0.15, label="Target Zone (70-100%)")
        ax2.set_xlabel("CPU Efficiency [%]\n(TotalCPU / [Elapsed * Cores])", fontweight="bold")
        ax2.set_ylabel("Number of Jobs", fontweight="bold")
        ax2.set_title(f"CPU Efficiency Distribution (reportseff N={len(valid_cpueff)})", fontweight="bold")
        ax2.legend(loc="upper left", frameon=True, fontsize=8.5)
        ax2.grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: Memory Efficiency Distribution
    valid_memeff = df[df["mem_efficiency_pct"] > 0]["mem_efficiency_pct"]
    if not valid_memeff.empty:
        ax3.hist(
            valid_memeff,
            bins=np.linspace(0, 100, 21),
            color="#8172B2",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.85
        )
        ax3.axvline(valid_memeff.median(), color="orange", linestyle="-", linewidth=2, label=f"Median ({valid_memeff.median():.1f}%)")
        ax3.axvspan(70, 85, color="forestgreen", alpha=0.15, label="Target Safe Zone (70-85%)")
        ax3.axvline(100, color="crimson", linestyle="--", linewidth=1.8, label="OOM Threshold (100%)")
        ax3.set_xlabel("Memory Efficiency [%]\n(MaxRSS / ReqMem)", fontweight="bold")
        ax3.set_ylabel("Number of Jobs", fontweight="bold")
        ax3.set_title(f"Memory Efficiency Distribution (N={len(valid_memeff)})", fontweight="bold")
        ax3.legend(loc="upper right", frameon=True, fontsize=8.5)
        ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
else:
    print("Matplotlib / Pandas not available in current environment; skipping graphical rendering.")"""))

    # Cell 10: Resource Allocation vs Actual Consumption Parity Analysis Markdown
    cells.append(md_cell(r"""### 5. Resource Request vs. Actual Consumption Parity Analysis

To eliminate queue waste, we compare requested allocations directly against consumed resources:
- **Requested Walltime vs. Consumed Walltime**: Points far below the 1:1 diagonal represent over-allocated walltime cushions that inflate job queue footprints.
- **Requested Memory vs. Consumed Peak RSS**: Points well below the diagonal represent idle allocated RAM. Target allocation falls within the 75%–85% parity corridor."""))

    # Cell 11: Parity Scatter Figure
    cells.append(code_cell(r"""if plt is not None and pd is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # Subplot 1: Requested Walltime vs Actual Walltime
    valid_time = df[(df["req_time_min"] > 0) & (df["used_walltime_sec"] > 0)].copy()
    if not valid_time.empty:
        used_time_min = valid_time["used_walltime_sec"] / 60.0
        ax1.scatter(
            valid_time["req_time_min"],
            used_time_min,
            c=valid_time["time_efficiency_pct"],
            cmap="viridis",
            s=110,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
            zorder=4
        )
        t_diag = np.logspace(0, 4, 100)
        ax1.plot(t_diag, t_diag, color="crimson", linestyle="--", linewidth=1.5, label="1:1 Parity (Timeout Boundary)")
        ax1.plot(t_diag, t_diag * 0.75, color="forestgreen", linestyle=":", linewidth=1.5, label="75% Target Utilization")

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Requested Walltime (--time) [Minutes]", fontweight="bold")
        ax1.set_ylabel("Actual Elapsed Walltime [Minutes]", fontweight="bold")
        ax1.set_title("Requested vs. Consumed Walltime Parity", fontweight="bold")
        ax1.legend(loc="upper left", frameon=True, fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Requested Memory vs Peak MaxRSS
    valid_mem_p = df[(df["req_mem_gb"] > 0) & (df["used_max_rss_gb"] > 0)].copy()
    if not valid_mem_p.empty:
        sc2 = ax2.scatter(
            valid_mem_p["req_mem_gb"],
            valid_mem_p["used_max_rss_gb"],
            c=valid_mem_p["mem_efficiency_pct"],
            cmap="plasma",
            s=110,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
            zorder=4
        )
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_label("Mem Efficiency (%)", fontsize=10)

        m_diag = np.logspace(0, 3, 100)
        ax2.plot(m_diag, m_diag, color="crimson", linestyle="--", linewidth=1.5, label="1:1 Parity (OOM Boundary)")
        ax2.plot(m_diag, m_diag * 0.80, color="forestgreen", linestyle=":", linewidth=1.5, label="80% Target Utilization")

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Requested Memory (--mem) [GB]", fontweight="bold")
        ax2.set_ylabel("Peak Consumed Memory (MaxRSS) [GB]", fontweight="bold")
        ax2.set_title("Requested vs. Consumed Physical Memory Parity", fontweight="bold")
        ax2.legend(loc="upper left", frameon=True, fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
else:
    print("Matplotlib / Pandas not available in current environment; skipping graphical rendering.")"""))

    # Cell 12: Efficiency by Job State & Slack Analysis Markdown
    cells.append(md_cell(r"""### 6. Efficiency Breakdown by Job State & Allocation Slack Analysis

Analyzing efficiency distributions by terminal Slurm state (`COMPLETED`, `TIMEOUT`, `FAILED`, `CANCELLED`) reveals operational dynamics:
- `COMPLETED` jobs average $<5\%$ time efficiency, indicating opportunities to reduce `--time` requests and obtain higher queue backfill priority.
- `TIMEOUT` jobs consistently exhibit `TimeEff` $\ge 100\%$, pinpointing simulations that exceeded requested limits.
- Memory slack ($(\mathrm{ReqMem} - \mathrm{MaxRSS}) \times \mathrm{Elapsed}$) quantifies reserved GB-hours left idle."""))

    # Cell 13: Job State Efficiency Boxplot & Slack Code
    cells.append(code_cell(r"""if plt is not None and pd is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

    # Subplot 1: Efficiency by Job State
    df["clean_state"] = df["state"].fillna("UNKNOWN").apply(lambda s: str(s).split()[0])
    top_states = ["COMPLETED", "TIMEOUT", "FAILED", "CANCELLED"]
    state_sub = df[df["clean_state"].isin(top_states) & (df["time_efficiency_pct"] > 0)]

    if not state_sub.empty:
        groups = [state_sub[state_sub["clean_state"] == st]["time_efficiency_pct"].values for st in top_states]
        valid_groups = [g for g in groups if len(g) > 0]
        valid_labels = [st for st, g in zip(top_states, groups) if len(g) > 0]

        if valid_groups:
            ax1.boxplot(valid_groups, tick_labels=valid_labels, patch_artist=True,
                        boxprops=dict(facecolor="#4C72B0", alpha=0.7),
                        medianprops=dict(color="red", linewidth=1.5))
            ax1.axhline(100, color="crimson", linestyle="--", linewidth=1.5, label="100% Time Limit")
            ax1.set_ylabel("Time Efficiency [%]", fontweight="bold")
            ax1.set_title("Time Efficiency by Job Terminal State", fontweight="bold")
            ax1.legend(loc="upper right", frameon=True, fontsize=9)
            ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Memory Efficiency by Job State
    state_mem = df[df["clean_state"].isin(top_states) & (df["mem_efficiency_pct"] > 0)]
    if not state_mem.empty:
        groups_m = [state_mem[state_mem["clean_state"] == st]["mem_efficiency_pct"].values for st in top_states]
        valid_groups_m = [g for g in groups_m if len(g) > 0]
        valid_labels_m = [st for st, g in zip(top_states, groups_m) if len(g) > 0]

        if valid_groups_m:
            ax2.boxplot(valid_groups_m, tick_labels=valid_labels_m, patch_artist=True,
                        boxprops=dict(facecolor="#55A868", alpha=0.7),
                        medianprops=dict(color="red", linewidth=1.5))
            ax2.axhline(80, color="forestgreen", linestyle="--", linewidth=1.5, label="80% Target Calibration")
            ax2.set_ylabel("Memory Efficiency [%]", fontweight="bold")
            ax2.set_title("Memory Efficiency by Job Terminal State", fontweight="bold")
            ax2.legend(loc="upper right", frameon=True, fontsize=9)
            ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
else:
    print("Matplotlib / Pandas not available in current environment; skipping graphical rendering.")"""))

    # Cell 14: Interactive Rightsizer Tool Markdown
    cells.append(md_cell(r"""### 7. Programmatic Slurm Resource Rightsizer

Use the utility function below to rightsize memory, walltime, and CPU allocations for any planned simulation run:
```python
calculate_rightsized_job(model="Bu2026Fixed", n_transients=2_000_000)
```"""))

    # Cell 15: Rightsizer Code
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

    # Cell 16: Suite-Specific Rightsizing Markdown
    cells.append(md_cell(r"""### 8. Suite-Specific Empirical Rightsizing & OOM Risk Analysis

For analysis suites with a statistically significant sample size ($N \ge 20$), we determine the optimal resource directives by reconciling:
1. **Empirical OOM Floor ($\mathrm{Mem}_{\mathrm{floor}}$)**: The highest memory tier where jobs crashed with `OUT_OF_MEMORY`. Allocations at or below this floor represent guaranteed or high-probability failure points.
2. **Completed Usage Upper Bound ($P_{95}(\mathrm{MaxRSS})$)**: The 95th percentile physical memory footprint among completed simulations.
3. **Optimal Safe Allocation**:
   $$\mathrm{RecMem} = \max\Big(\mathrm{Mem}_{\mathrm{floor}} \times 1.25,\; P_{95}(\mathrm{MaxRSS}) \times 1.25\Big)$$
   mapped to the nearest standard Slurm memory tier (e.g., 32 GB, 64 GB, 128 GB, 256 GB) to achieve **70%–85%** memory efficiency while eliminating OOM kills."""))

    # Cell 17: Suite Rightsizing Visualizations & Summary Table
    cells.append(code_cell(r"""if plt is not None and pd is not None:
    from telemetry.analyzer import TelemetryAnalyzer
    from telemetry.harvester import JobHarvester, infer_suite_from_job_name

    # Ensure suite is populated
    df["clean_suite"] = df.apply(
        lambda row: row["suite"] if str(row.get("suite", "")).strip() not in ("", "None", "Unknown")
        else infer_suite_from_job_name(str(row.get("job_name", ""))),
        axis=1
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))

    # Subplot 1: OOM Failure Probability vs Requested Memory for Significant Suites
    eval_suites = ["Suite 02: Granular Criteria Sweep", "Suite 01: Large-Scale Rate Limits"]
    colors_suite = {"Suite 02: Granular Criteria Sweep": "#C44E52", "Suite 01: Large-Scale Rate Limits": "#4C72B0"}

    for s_name in eval_suites:
        s_df = df[df["clean_suite"] == s_name].copy()
        if len(s_df) >= 20:
            oom_rates = []
            mem_tiers = sorted([m for m in s_df["req_mem_gb"].dropna().unique() if m > 0])
            for m in mem_tiers:
                tier_jobs = s_df[s_df["req_mem_gb"] == m]
                oom_cnt = len(tier_jobs[tier_jobs["state"].str.startswith("OUT_OF_MEMORY")])
                rate = (oom_cnt / len(tier_jobs)) * 100.0 if len(tier_jobs) > 0 else 0.0
                oom_rates.append((m, rate, len(tier_jobs)))

            tiers_x = [pt[0] for pt in oom_rates if pt[2] >= 5]
            rates_y = [pt[1] for pt in oom_rates if pt[2] >= 5]
            if tiers_x:
                label_short = s_name.split(":")[0]
                ax1.plot(tiers_x, rates_y, marker="o", linewidth=2.2, color=colors_suite.get(s_name, "gray"),
                         label=f"{label_short} (N={len(s_df):,})")

    ax1.set_xscale("log")
    ax1.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.axhline(0, color="forestgreen", linestyle=":", alpha=0.8, label="0% Failure Target")
    ax1.axvline(32, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="Historical 32 GB OOM Cliff")
    ax1.set_xlabel("Requested Memory Allocation (--mem) [GB]", fontweight="bold")
    ax1.set_ylabel("Empirical OOM Failure Rate [%]", fontweight="bold")
    ax1.set_title("OOM Failure Probability vs. Allocated Memory Tier", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: MaxRSS for Completed Runs vs Allocation Ceiling
    suite02_comp = df[(df["clean_suite"] == "Suite 02: Granular Criteria Sweep") & (df["state"].str.startswith("COMPLETED"))]
    valid_rss = suite02_comp[suite02_comp["used_max_rss_gb"] > 0]["used_max_rss_gb"]
    if not valid_rss.empty:
        ax2.hist(valid_rss, bins=np.linspace(0, 50, 26), color="#55A868", edgecolor="black", linewidth=0.7, alpha=0.85)
        ax2.axvline(32, color="crimson", linestyle="--", linewidth=2, label="Historical Alloc (32 GB) -> 32.1% OOM")
        ax2.axvline(valid_rss.quantile(0.95), color="orange", linestyle="-", linewidth=2, label=f"Completed P95 RSS ({valid_rss.quantile(0.95):.1f} GB)")
        ax2.axvline(64, color="forestgreen", linestyle="-.", linewidth=2, label="Recommended Alloc (64 GB) -> 0% OOM")
        ax2.set_xlabel("Peak Physical RAM Consumed (MaxRSS) [GB]", fontweight="bold")
        ax2.set_ylabel("Number of Completed Runs", fontweight="bold")
        ax2.set_title("Suite 02 Completed MaxRSS vs. OOM Failure Cliff", fontweight="bold")
        ax2.legend(loc="upper right", frameon=True, fontsize=8.5)
        ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()

    # Programmatic Suite Rightsizing Table
    root = Path("..") if Path("../telemetry").exists() else Path(".")
    tracker = JobTelemetryTracker(project_root=root)
    analyzer = TelemetryAnalyzer(tracker.load_registry())
    suite_recs = analyzer.analyze_suite_rightsizing(min_jobs=20)

    print("\n" + "=" * 90)
    print("EMPIRICAL SUITE RIGHTSIZING TABLE (Statistically Significant Suites N >= 20)")
    print("=" * 90)
    for s_name, data in suite_recs.items():
        print(f"\nSuite: {s_name}")
        print(f"  Tracked Jobs:          {data['sample_size']:,} ({data['completed_count']:,} completed, {data['oom_count']:,} OOM kills)")
        print(f"  Historical OOM Rate:   {data['oom_failure_rate_pct']:.1f}%")
        print(f"  OOM Memory Floor:      {data['oom_memory_floor_gb']:.1f} GB (crashed at or below this allocation)")
        print(f"  Completed Peak RSS:    Median {data['completed_rss_p50_gb']:.2f} GB | P95 {data['completed_rss_p95_gb']:.2f} GB | Max {data['completed_rss_max_gb']:.2f} GB")
        print(f"  Completed Walltime:    Median {data['completed_walltime_p50_min']:.1f} min | P95 {data['completed_walltime_p95_min']:.1f} min")
        print(f"  RECOMMENDED SBATCH DIRECTIVES:")
        print(f"    #SBATCH -p {data['recommended_partition']}")
        print(f"    #SBATCH --mem={int(data['recommended_mem_gb'])}g")
        print(f"    #SBATCH --time={data['recommended_walltime_str']}")
        print(f"    #SBATCH --cpus-per-task={data['recommended_cpus']}")
        print(f"  Queue Advice:          {data['queue_advice']}")
        print(f"  Mitigation Notes:      {data['risk_notes']}")
        print("-" * 90)
else:
    print("Matplotlib / Pandas not available in current environment; skipping graphical rendering.")"""))

    # Cell 18: Suite Rightsizer Function Code
    cells.append(code_cell(r"""def calculate_suite_rightsized_job(suite_name: str):
    \"\"\"Interactive utility to lookup empirical rightsizing directives for an analysis suite.\"\"\"
    root = Path("..") if Path("../telemetry").exists() else Path(".")
    tracker = JobTelemetryTracker(project_root=root)
    analyzer = TelemetryAnalyzer(tracker.load_registry())
    suite_recs = analyzer.analyze_suite_rightsizing(min_jobs=5)

    match = next((v for k, v in suite_recs.items() if suite_name.lower() in k.lower()), None)
    label = next((k for k, v in suite_recs.items() if suite_name.lower() in k.lower()), suite_name)

    if not match:
        print(f"No statistically significant data found for '{suite_name}'. Available: {list(suite_recs.keys())}")
        return

    print("=" * 72)
    print(f"SLURM EMPIRICAL RIGHTSIZING: {label}")
    print("=" * 72)
    print(f"Sample Size:             {match['sample_size']:,} jobs ({match['completed_count']:,} completed, {match['oom_count']:,} OOM kills)")
    print(f"Historical OOM Rate:     {match['oom_failure_rate_pct']:.1f}%")
    print(f"Historical OOM Mem Floor:{match['oom_memory_floor_gb']:.1f} GB (crashed at/below this allocation)")
    print(f"Completed Peak RAM (P95):{match['completed_rss_p95_gb']:.2f} GB (Max: {match['completed_rss_max_gb']:.2f} GB)")
    print(f"Completed Walltime (P95):{match['completed_walltime_p95_min']:.1f} minutes")
    print("-" * 72)
    print("RECOMMENDED SLURM SBATCH DIRECTIVES:")
    print(f"  #SBATCH -p {match['recommended_partition']}")
    print(f"  #SBATCH --mem={int(match['recommended_mem_gb'])}g")
    print(f"  #SBATCH --time={match['recommended_walltime_str']}")
    print(f"  #SBATCH --cpus-per-task={match['recommended_cpus']}")
    print("-" * 72)
    print("Queue Placement Strategy:")
    print(f"  {match['queue_advice']}")
    print("Risk & Failure Mitigation Basis:")
    print(f"  {match['risk_notes']}")
    print("=" * 72)

# Demonstration: Rightsizing Suite 02 (Granular Criteria Sweep)
calculate_suite_rightsized_job("Suite 02")"""))

    # Write notebook file
    nb_obj = make_notebook(cells)
    out_path = Path("notebooks/job_resource_usage_analysis.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb_obj, indent=2))
    print(f"Wrote updated notebook to {out_path}")


if __name__ == "__main__":
    main()

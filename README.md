# ZTF 8-Year Kilonova Rate Limits

Injection–recovery simulations of kilonovae (KNe) in the full Zwicky Transient Facility observing history
(March 2018 – March 2026) to derive volumetric rate upper limits, using the
[`survey_sim`](../survey-sim) engine (fiesta POSSIS surrogate — labelled `Bu2026` here — and a one-zone Metzger model).

> **Catching up?** Open `notebooks/00_overview.ipynb` (every headline number, where it comes from, and every open
> item) and [`docs/decisions.md`](docs/decisions.md) (what was decided and why).
>
> **Background:** [`docs/review_2026-11.md`](docs/review_2026-11.md) — scientific and code review (Nov 2026)
> listing which results are trustworthy, which suites must be re-run, and what the manuscript draft must change.
> In short: Suites 01 and 07 are valid fixed-parameter ensembles; Suite 02's filter has been fixed and its candidate ensemble
> was re-simulated (100 compatible sets per model, Nov 2026); Suite 03 now samples a real population and must be re-run;
> the former Suites 04–06 did not run simulations and were retired; the limits assume (but the repository does not
> perform) a real-data search with zero candidates.

All numbers quoted anywhere in this project should come from the notebooks, which compute them from the
results snapshot — nothing in this README is a result.

---

## 1. Repository layout

```
kn-rates/
├── knrates/                      # shared analysis package (used by all notebooks)
│   ├── collect.py                #   builds the results snapshot from results/  (python -m knrates.collect)
│   ├── snapshot.py               #   loads a snapshot: latest, time-stamped, or live
│   ├── stats.py                  #   Poisson limits, pooled MC estimators, hypergeometric sub-sampling
│   ├── analysis.py               #   pooled limits, VT vs. epoch, VT vs. inclination, convergence
│   ├── literature.py             #   published limits/parameters with arXiv sources (verified)
│   ├── plotting.py               #   AAS-sized publication style, fixed colours per model
│   ├── figures.py                #   every figure, one function each (used by notebooks and the paper)
│   └── notebook.py               #   one-call notebook setup (snapshot + model selection + style)
├── snapshots/                    # VERSION-CONTROLLED compiled results (single ZIP per snapshot)
├── analysis_suites/
│   ├── 01_large_scale_rate_limits/   # 100 x 1e6 injections per fiducial model
│   ├── 02_granular_criteria_sweep/   # simulate-once, filter-many criteria sweep (see review: needs fixes)
│   ├── 03_prior_vs_at2017gfo/        # criteria regimes x populations (BroadPrior needs redesign)
│   └── 07_gw170817_rate_limits/      # 100 x 1e6 injections per GW170817-informed model
├── notebooks/                    # analysis notebooks, ordered by the paper's argument (start at 00_overview)
├── scripts/                      # Slurm submission / aggregation scripts (MSI)
├── telemetry/                    # Slurm job telemetry, classification, FairShare accounting
├── paper/                        # manuscript drafts, figures (written by the paper notebook)
├── tests/                        # pytest suite
└── defunct/                      # retired scripts, notebooks, generators and fabricated-output suites
```

`results/` and `logs/` are not version-controlled (tens of GB). They live on MSI; a copy can be unpacked
locally from `results-logs.tar.gz`.

## 2. Results snapshot (work anywhere, keep results under version control)

`python -m knrates.collect` reads every suite under `results/`, validates it and writes **one file**,
`snapshots/knrates_results_latest.zip` (≈6 MB: one CSV per table + `manifest.json` with provenance, git commit,
source fingerprints, checksums and warnings). Commit it after new runs.

```bash
python -m knrates.collect                               # update the 'latest' snapshot
python -m knrates.collect --timestamped                 # write snapshots/knrates_results_<UTC>.zip (never overwritten)
python -m knrates.collect --timestamped --also-latest   # both
python -m knrates.collect --dry-run                     # validate only
python -m knrates.collect --force                       # ignore the per-suite cache
```

Safety: tables are validated (schema, `n_det ≤ n_sim`, finite VT, efficiency consistency, duplicate tasks,
per-model run counts must not shrink vs. the previous snapshot, …). Any error aborts **without touching the
existing snapshot**; truncated result files from crashed jobs are skipped and listed in the manifest; the archive is
written to a temp file, verified and atomically renamed; concurrent updates are serialised with a file lock.
Unchanged suites are re-used from the previous snapshot, so routine updates are fast.

Every aggregation/filter job calls `scripts/update_results_snapshot.sh` at the end
(`SNAPSHOT_TIMESTAMPED=1` also keeps a dated copy). To add a new suite, add a `collect_<suite>()` function in
`knrates/collect.py` (see its docstring).

**In notebooks** the first cell selects the snapshot:

```python
SNAPSHOT = "latest"   # or "20261109T184311Z" / a unique prefix like "20261109" / "newest" / a .zip path / "live"
```

## 3. Notebooks

**Start with `notebooks/00_overview.ipynb`.** It lists what the snapshot contains, every headline number with the
notebook and paper section it comes from, and every open `\tbd{}` in the manuscripts. Decisions and their
reasons are in [`docs/decisions.md`](docs/decisions.md).

| Notebook | Question | Paper |
|---|---|---|
| `00_overview` | What is in the snapshot, what the results say, and what is still open | — |
| `01_limits` | ⟨VT⟩ and rate limits; distance reach | §4, Figs. 3–4, Table 1 |
| `02_survey_time_and_previous_searches` | Sensitivity over the survey; comparison with Andreoni+2020/2021 | §4.1, Fig. 1 |
| `03_viewing_angle` | How much the orientation matters | §5, Fig. 2 |
| `04_systematics` | MC noise, z_max, parameter set (Suite 07), light curves vs. AT2017gfo, criteria, population (Suite 03) | §6, §8, Fig. 5 |
| `05_criteria_companion` | Response of ⟨VT⟩ to each selection criterion (Suite 02) | companion paper, Fig. 6 |
| `paper_figures` | Writes `paper/figures/` and Table 1 (`paper/table1_master_rate_limits.{csv,tex}`) | all figures |
| `telemetry_job_resources` | Slurm telemetry; `ANALYSIS_SCOPE = "numbered" \| "preliminary" \| "all"`, FairShare billing | — |

Each figure is drawn by one function in `knrates/figures.py`; `knrates.notebook.setup()` loads the snapshot and
keeps only the Suite 01 POSSIS models unless `include_metzger` / `include_suite07` are set. Each notebook gives a
one-line computed **Result** under each figure; the longer per-figure explanations are in the archived per-suite
notebooks, `defunct/notebooks_by_suite/`, which still run.

Light-curve tables used by notebooks 03–04: `python scripts/make_model_lightcurves.py` (survey_sim env) and
`python scripts/make_metzger_check.py` (needs cargo; also adds the AT2017gfo photometry), then
`python -m knrates.collect --suites models`.

Run all notebooks: `python execute_notebooks.py` (or a subset by path). Executing notebooks needs a local kernel
port; inside the Claude Code sandbox enable `sandbox.network.allowLocalBinding`.

## 4. Running the simulations (MSI)

```bash
conda activate kn_rates_dev            # or simsurvey_no_gpu (scripts auto-detect)
bash scripts/submit_suite01_rate_limits.sh      # Suite 01 ensemble + aggregation (+ snapshot)
bash scripts/submit_suite07_rate_limits.sh      # Suite 07 ensemble + aggregation (+ snapshot)
bash scripts/submit_fast_criteria_sweep.sh --fiducial   # Suite 02 candidates + filter (+ snapshot)
sbatch scripts/03_run_prior_vs_at2017gfo_comparison.sh  # Suite 03 (+ snapshot)
./scripts/submit_all_suites.sh                  # everything with dependencies
```

Local smoke test: `python analysis_suites/01_large_scale_rate_limits/run_large_scale_simulation.py --model Bu2026Fixed --n-transients 100000`.

## 5. Job telemetry & FairShare

- Harvest (on MSI): `python -m telemetry update` — merges sacct/reportseff/logs into
  `telemetry/job_resource_registry.csv` and, if `sshare` is available, appends a FairShare snapshot to
  `telemetry/fairshare_snapshots.csv` (`python -m telemetry fairshare` for a snapshot only).
- Analyse (anywhere, CSV only): `notebooks/telemetry_job_resources.ipynb`; set `ANALYSIS_SCOPE`
  (`"numbered"` production suites 01–07, `"preliminary"` older exploratory runs, `"all"`), `INCLUDE_OTHER`,
  `DATE_RANGE`, `MEM_PER_CORE_GB` in the first cell. Covers job outcomes, queue waits, efficiency, billed usage
  (max of cores and memory/3.9 GB — an assumption to confirm per partition), wasted and failed allocation,
  FairShare-aware `#SBATCH` rightsizing and queue-wait vs. recent-usage correlation.
- Classification: `telemetry/classify.py`; cleaned/de-duplicated table: `telemetry.frame.load_registry_frame()`.
- Hourly automation: `./scripts/install_cron.sh --install` (scrontab) or `./scripts/run_telemetry_daemon.sh start`.

## 6. Tests

```bash
python -m pytest -q tests/        # tests/test_suite07_gw170817.py additionally needs survey_sim + jax
```

"""Unit tests for the job telemetry, queue dynamics, and resource rightsizing system."""

import os
from pathlib import Path
import tempfile
import unittest

from telemetry.analyzer import TelemetryAnalyzer
from telemetry.harvester import JobHarvester, parse_memory_to_gb, parse_percentage, parse_slurm_time_to_seconds
from telemetry.models import JobTelemetryRecord, RightsizingRecommendation
from telemetry.passive_logger import JobResourceTracker, get_cpu_seconds, get_peak_rss_gb
from telemetry.tracker import JobTelemetryTracker


class TestTelemetryUnit(unittest.TestCase):
    """Test unit conversions, parsing functions, and model methods."""

    def test_memory_parsing(self):
        self.assertEqual(parse_memory_to_gb("750G"), 750.0)
        self.assertEqual(parse_memory_to_gb("32g"), 32.0)
        self.assertAlmostEqual(parse_memory_to_gb("3900M"), 3.809, places=2)
        self.assertAlmostEqual(parse_memory_to_gb("1048576K"), 1.0, places=2)
        self.assertEqual(parse_memory_to_gb("0"), 0.0)
        self.assertEqual(parse_memory_to_gb(""), 0.0)

    def test_percentage_parsing(self):
        self.assertEqual(parse_percentage("123.3%"), 123.3)
        self.assertEqual(parse_percentage("15.0%"), 15.0)
        self.assertEqual(parse_percentage("0.2%"), 0.2)
        self.assertEqual(parse_percentage("---"), 0.0)
        self.assertEqual(parse_percentage("N/A"), 0.0)
        self.assertEqual(parse_percentage(""), 0.0)

    def test_slurm_time_parsing(self):
        self.assertEqual(parse_slurm_time_to_seconds("01:00:00"), 3600.0)
        self.assertEqual(parse_slurm_time_to_seconds("20:47:00"), 74820.0)
        self.assertEqual(parse_slurm_time_to_seconds("1-02:00:00"), 93600.0)
        self.assertEqual(parse_slurm_time_to_seconds("02:30"), 150.0)
        self.assertEqual(parse_slurm_time_to_seconds("45"), 45.0)
        self.assertEqual(parse_slurm_time_to_seconds("---"), 0.0)
        self.assertEqual(parse_slurm_time_to_seconds(""), 0.0)

    def test_record_derived_metrics(self):
        rec = JobTelemetryRecord(
            job_id="12345",
            submit_time="2026-10-20 14:00:00",
            start_time="2026-10-20 14:15:00",
            end_time="2026-10-20 15:15:00",
            req_cpus=4,
            req_mem_gb=32.0,
            req_time_min=120.0,
            used_max_rss_gb=16.0,
            used_cpu_sec=7200.0,
        )
        rec.compute_derived_metrics()

        self.assertEqual(rec.submit_hour, 14)
        self.assertEqual(rec.submit_day_of_week, "Tue")
        self.assertFalse(rec.is_weekend)
        self.assertEqual(rec.queue_wait_sec, 900.0)
        self.assertEqual(rec.queue_wait_min, 15.0)
        self.assertEqual(rec.used_walltime_sec, 3600.0)
        self.assertEqual(rec.mem_efficiency_pct, 50.0)
        # 7200 CPU sec / (3600 wall sec * 4 cpus) = 50%
        self.assertEqual(rec.cpu_efficiency_pct, 50.0)
        # 3600 wall sec / (120 min * 60) = 50%
        self.assertEqual(rec.time_efficiency_pct, 50.0)
        # Wasted core hours: 4 cpus * 1 hr - 2 CPU hrs = 2 core-hrs
        self.assertAlmostEqual(rec.wasted_core_hours, 2.0, places=2)
        # Wasted mem slack: (32GB - 16GB) * 1 hr = 16 GB-hrs
        self.assertAlmostEqual(rec.wasted_mem_gb_hours, 16.0, places=2)

    def test_reportseff_table_parsing(self):
        sample_output = """
 JobID      State         Elapsed  TimeEff   CPUEff   MemEff 
              4735813     FAILED        00:11:51   9.9%     38.5%    28.6%
            4477571_0     FAILED        00:01:19   1.1%     24.1%    15.9% 
              4735752     TIMEOUT       00:01:14  123.3%     2.7%     2.2%  
              5118224     PENDING            ---    ---      ---      ---
        """
        harvester = JobHarvester()
        records = harvester.parse_reportseff_output(sample_output)
        self.assertEqual(len(records), 4)

        r1 = next(r for r in records if r.job_id == "4735813")
        self.assertEqual(r1.state, "FAILED")
        self.assertEqual(r1.used_walltime_sec, 711.0)
        self.assertEqual(r1.time_efficiency_pct, 9.9)
        self.assertEqual(r1.cpu_efficiency_pct, 38.5)
        self.assertEqual(r1.mem_efficiency_pct, 28.6)
        self.assertAlmostEqual(r1.req_time_min, 119.7, delta=1.0)

        # Check array task parsing
        r2 = next(r for r in records if r.job_id == "4477571" and r.array_task_id == "0")
        self.assertEqual(r2.state, "FAILED")
        self.assertEqual(r2.time_efficiency_pct, 1.1)
        self.assertEqual(r2.cpu_efficiency_pct, 24.1)

        # Check timeout > 100%
        r3 = next(r for r in records if r.job_id == "4735752")
        self.assertEqual(r3.state, "TIMEOUT")
        self.assertEqual(r3.time_efficiency_pct, 123.3)

    def test_reportseff_pipe_parsing(self):
        sample_pipe = """JobID|State|Elapsed|TimeEff|CPUEff|MemEff
4735813|FAILED|00:11:51|9.9%|38.5%|28.6%
5016210|TIMEOUT|02:00:39|100.5%|34.8%|14.4%
"""
        harvester = JobHarvester()
        records = harvester.parse_reportseff_output(sample_pipe)
        self.assertEqual(len(records), 2)
        r_timeout = records[1]
        self.assertEqual(r_timeout.job_id, "5016210")
        self.assertEqual(r_timeout.state, "TIMEOUT")
        self.assertEqual(r_timeout.time_efficiency_pct, 100.5)
        self.assertEqual(r_timeout.cpu_efficiency_pct, 34.8)
        self.assertEqual(r_timeout.mem_efficiency_pct, 14.4)

    def test_reportseff_merge_overwrites_sacct_cpueff(self):
        sacct_rec = JobTelemetryRecord(
            job_id="4735813",
            req_cpus=4,
            req_mem_gb=32.0,
            used_walltime_sec=711.0,
            used_cpu_sec=2844.0,  # CPUTimeRAW allocated
            cpu_efficiency_pct=100.0,  # Sacct bug artifact
            mem_efficiency_pct=100.0,
            state="COMPLETED",
            data_source="sacct",
        )
        reportseff_rec = JobTelemetryRecord(
            job_id="4735813",
            used_walltime_sec=711.0,
            time_efficiency_pct=9.9,
            cpu_efficiency_pct=38.5,
            mem_efficiency_pct=28.6,
            state="FAILED",
            data_source="reportseff",
        )
        harvester = JobHarvester()
        harvester._merge_records(sacct_rec, reportseff_rec)

        self.assertEqual(sacct_rec.cpu_efficiency_pct, 38.5)
        self.assertEqual(sacct_rec.time_efficiency_pct, 9.9)
        self.assertEqual(sacct_rec.mem_efficiency_pct, 28.6)
        self.assertEqual(sacct_rec.state, "FAILED")
        self.assertEqual(sacct_rec.data_source, "reportseff")

    def test_analyzer_efficiencies(self):
        records = [
            JobTelemetryRecord(job_id="1", time_efficiency_pct=10.0, cpu_efficiency_pct=25.0, mem_efficiency_pct=50.0, state="COMPLETED", used_walltime_sec=100.0, req_cpus=2),
            JobTelemetryRecord(job_id="2", time_efficiency_pct=100.5, cpu_efficiency_pct=80.0, mem_efficiency_pct=90.0, state="TIMEOUT", used_walltime_sec=200.0, req_cpus=4),
        ]
        analyzer = TelemetryAnalyzer(records)
        eff = analyzer.analyze_efficiencies()

        self.assertEqual(eff["time_efficiency"]["count"], 2)
        self.assertAlmostEqual(eff["time_efficiency"]["median"], 55.25, places=2)
        self.assertIn("COMPLETED", eff["by_state"])
        self.assertIn("TIMEOUT", eff["by_state"])

    def test_harvester_benchmarks(self):
        harvester = JobHarvester()
        benchmarks = harvester.harvest_known_benchmarks()
        self.assertGreaterEqual(len(benchmarks), 4)

        agate_10m = next((b for b in benchmarks if b.n_transients == 10000000), None)
        self.assertIsNotNone(agate_10m)
        self.assertEqual(agate_10m.job_id, "14237734")
        self.assertEqual(agate_10m.used_max_rss_gb, 649.67)
        self.assertEqual(agate_10m.used_walltime_sec, 74820.0)

    def test_scaling_fit_and_rightsizing(self):
        harvester = JobHarvester()
        records = harvester.harvest_known_benchmarks()
        for r in records:
            r.compute_derived_metrics()

        analyzer = TelemetryAnalyzer(records)
        scaling = analyzer.fit_resource_scaling()
        self.assertIn("Global", scaling)
        global_fit = scaling["Global"]
        self.assertGreater(global_fit["ram_gb_per_million"], 0)
        self.assertGreater(global_fit["walltime_sec_per_million"], 0)

        # Test rightsizing recommendation for 1M sources
        rec_1m = analyzer.recommend_resources("Bu2026Fixed", 1_000_000)
        self.assertIsInstance(rec_1m, RightsizingRecommendation)
        self.assertGreaterEqual(rec_1m.recommended_mem_gb, 32.0)
        self.assertGreater(rec_1m.recommended_walltime_sec, 0)
        self.assertEqual(rec_1m.recommended_partition, "msismall")

        # Test rightsizing for 10M sources
        rec_10m = analyzer.recommend_resources("Bu2026Vary", 10_000_000)
        self.assertEqual(rec_10m.recommended_mem_gb, 750.0)
        self.assertIn("HIGH MEMORY", rec_10m.queue_advice)

    def test_passive_logger_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events_file = Path(tmpdir) / "test_events.jsonl"
            out_file = Path(tmpdir) / "output.txt"
            out_file.write_text("simulated output data")

            with JobResourceTracker(
                job_name="test_passive",
                model="Metzger",
                n_transients=50000,
                output_path=out_file,
                events_file=events_file,
            ):
                # Simulate work
                total = sum(i * i for i in range(100_000))
                self.assertGreater(total, 0)

            self.assertTrue(events_file.exists())
            content = events_file.read_text().strip()
            self.assertIn("test_passive", content)
            self.assertIn("Metzger", content)

    def test_tracker_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmproot = Path(tmpdir)
            tracker = JobTelemetryTracker(project_root=tmproot)
            recs = tracker.update()
            self.assertGreater(len(recs), 0)
            self.assertTrue(tracker.registry_path.exists())
            self.assertTrue(tracker.scaling_path.exists())

            report = tracker.generate_report()
            self.assertIn("HPC Job Telemetry", report)
            self.assertIn("Memory Tier", report)
            self.assertIn("Rightsizing Recommendations", report)

    def test_suite_rightsizing_tri_resource(self):
        """Test empirical rightsizing across RAM, walltime, and CPU cores."""
        # Create a mock suite with 25 completed jobs (low CPU efficiency, high memory) and 5 OOM kills
        records = []
        for i in range(25):
            records.append(JobTelemetryRecord(
                job_id=f"comp_{i}",
                suite="Suite 99: Test Suite",
                state="COMPLETED",
                req_mem_gb=64.0,
                used_max_rss_gb=40.0 + (i % 5),  # 40-44 GB
                req_cpus=8,
                used_walltime_sec=1800.0,  # 30 min
                used_cpu_sec=1440.0,       # 10% CPUEff -> 0.8 effective cores
                cpu_efficiency_pct=10.0,
                req_time_min=120.0,
            ))
        for j in range(5):
            records.append(JobTelemetryRecord(
                job_id=f"oom_{j}",
                suite="Suite 99: Test Suite",
                state="OUT_OF_MEMORY",
                req_mem_gb=32.0,  # OOM floor = 32 GB
                used_max_rss_gb=32.0,
                req_cpus=8,
                req_time_min=120.0,
            ))
        for k in range(2):
            records.append(JobTelemetryRecord(
                job_id=f"timeout_{k}",
                suite="Suite 99: Test Suite",
                state="TIMEOUT",
                req_mem_gb=64.0,
                used_walltime_sec=3600.0,  # 60 min timeout
                req_time_min=60.0,
            ))

        analyzer = TelemetryAnalyzer(records)
        suite_recs = analyzer.analyze_suite_rightsizing(min_jobs=10)
        self.assertIn("Suite 99: Test Suite", suite_recs)
        rec = suite_recs["Suite 99: Test Suite"]

        # Check memory rightsizing: OOM floor is 32GB, P95 RSS ~44GB -> recommended 64GB
        self.assertEqual(rec["oom_memory_floor_gb"], 32.0)
        self.assertEqual(rec["recommended_mem_gb"], 64.0)

        # Check walltime rightsizing: timeout occurred at 60 min -> recommended must be > 60 min (clears timeout floor)
        self.assertGreater(rec["recommended_walltime_sec"], 3600.0)
        self.assertGreaterEqual(rec["timeout_floor_min"], 60.0)

        # Check CPU rightsizing: 8 cores allocated but effective usage is ~0.8 cores
        # Recommended CPUs should be rightsized (e.g. 4 due to 64GB NUMA ratio)
        self.assertLessEqual(rec["recommended_cpus"], 4)
        self.assertGreaterEqual(rec["concurrency_multiplier"], 2.0)
        self.assertGreater(rec["su_savings_pct"], 0.0)

    def test_notebook_cell_ids_validity(self):
        """Verify that all notebook cells contain valid string IDs to prevent PR diff errors."""
        import json
        nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "job_resource_usage_analysis.ipynb"
        self.assertTrue(nb_path.exists())
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for idx, cell in enumerate(nb["cells"]):
            cell_id = cell.get("id")
            self.assertIsNotNone(cell_id, f"Cell {idx} has None id (violates nbformat 4.5 schema)")
            self.assertIsInstance(cell_id, str, f"Cell {idx} id is not a string: {cell_id}")
            self.assertGreater(len(cell_id), 0, f"Cell {idx} id is empty")



try:  # the analysis layer needs pandas; the harvester side does not
    import numpy as np
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:  # pragma: no cover
    HAVE_PANDAS = False

from telemetry.classify import NUMBERED_EPOCH, classify_job, canonical_suite_name, parse_suite_label
from telemetry.fairshare import capture_sshare_snapshot, parse_sshare_output
from telemetry.harvester import infer_suite_from_job_name
from telemetry.passive_logger import slurm_env_mem_gb


class TestClassification(unittest.TestCase):
    """Canonical job classification (telemetry.classify)."""

    AFTER = "2026-10-28T20:28:19"
    BEFORE = "2026-08-17T18:00:36"

    def check(self, name, category, suite_id=None, suite=None, when=AFTER, **kw):
        jc = classify_job(name, suite=suite, when=when, **kw)
        self.assertEqual(jc.category, category, f"{name!r} -> {jc}")
        self.assertEqual(jc.suite_id, suite_id, f"{name!r} -> {jc}")
        return jc

    def test_numbered_suites_by_job_name(self):
        cases = {
            "kn_rate_limits_1M_ensemble": "01", "kn_rate_limits_10M": "01", "kn_agg_1M_dist": "01",
            "kn_agg_suite01": "01", "kn_criteria_sweep": "02", "kn_crit_ultrafine": "02",
            "kn_crit_cand": "02", "kn_crit_filter": "02", "kn_crit_audit": "02",
            "kn_prior_vs_at2017gfo": "03", "kn_andreoni_comp": "04", "kn_sample_convergence": "05",
            "kn_inclination_study": "06", "kn_gw170817_fixed_ensemble": "07", "kn_agg_suite07": "07",
        }
        for name, sid in cases.items():
            jc = self.check(name, "numbered", sid)
            self.assertTrue(jc.group.startswith(f"Suite {sid}: "))

    def test_inconsistent_suite_labels_normalize(self):
        for raw in ("Suite 05", "Suite 05: Sample Convergence", "suite 5", "SUITE 05:"):
            self.assertEqual(parse_suite_label(raw), ("05", "Sample Convergence"))
        a = classify_job("", suite="Suite 01")
        b = classify_job("", suite="Suite 01: Large-Scale Rate Limits")
        self.assertEqual(a.group, b.group)

    def test_new_suite_is_picked_up(self):
        jc = self.check("kn_new_thing", "numbered", "08", suite="Suite 08: Brand New Study")
        self.assertEqual(jc.group, "Suite 08: Brand New Study")
        jc = self.check("kn_other_thing", "numbered", "09", log_folder="logs/09_cool_new_suite")
        self.assertEqual(jc.suite_label, "Cool New Suite")
        self.check("kn_agg_suite12", "numbered", "12")
        jc = self.check("kn_unknown_future_job", "numbered", None)
        self.assertEqual(jc.group, "Unassigned kn_* jobs")

    def test_preliminary_overrides_legacy_suite_column(self):
        # the legacy harvester labelled these as numbered suites
        self.check("sim_msismall_1M_multi_marathon_array_1.sh", "preliminary",
                   suite="Suite 01: Large-Scale Rate Limits", when=self.BEFORE)
        self.check("run_detection_criteria_sweep.sh", "preliminary", suite="Suite 02: Granular Criteria Sweep",
                   when="2026-10-08T16:03:56")
        # randomized-parameter exploration, even though it ran after the epoch
        jc = self.check("kn_sim_1M_random", "preliminary", suite="Suite 01: Large-Scale Rate Limits")
        self.assertEqual(jc.group, "Randomized-parameter exploration")
        self.check("submit_1M_randomized_suite01.sh", "preliminary")
        self.check("kn_sub_1M_rand", "preliminary")
        for name in ("sim.sh", "sim_bigmem_10M.sh", "sim_msismall_1M_debug.sh", "sim_msismall_100K_test.sh",
                     "run_prior_response_array.sh", "run_prior_sensitivity_array.sh", "run_rate_limit.sh"):
            self.check(name, "preliminary", when=self.BEFORE)

    def test_date_guard(self):
        # a kn_* name seeded from a submission log for a job that ran before the pipeline existed
        jc = self.check("kn_criteria_sweep", "preliminary", suite="Suite 02",
                        when=["2026-10-21 20:08:43", "2026-10-19T16:35:21", "None"])
        self.assertTrue(jc.rule.startswith("pre_epoch"))
        self.check("kn_criteria_sweep", "numbered", "02", when=NUMBERED_EPOCH)

    def test_other_and_orchestration(self):
        for name, part in (("bash", "interactive"), ("interactive", "msismall"), ("wrap", "msismall"),
                           ("ondemand/sys/dashboard/sys/bc_msi_jupyter", "interactive-gpu"),
                           ("anything", "interactive-long")):
            self.check(name, "other", partition=part)
        jc = self.check("/users/8/x/kn-rates/scripts/cron_update_telemetry.sh", "other")
        self.assertEqual(jc.group, "Telemetry infrastructure")
        for name in ("submit_all_suites.sh", "kn_submit_all", "kn_all_suites_96hr"):
            jc = self.check(name, "numbered", None)
            self.assertEqual(jc.group, "Pipeline orchestration")

    def test_seeded_benchmarks_and_fallbacks(self):
        self.check("sim_msismall_1M.sh", "preliminary", suite="Suite 05", data_source="empirical_benchmark")
        jc = self.check("whatever", "preliminary", job_id="14237734")
        self.assertEqual(jc.group, "Seeded reference benchmarks")
        self.check("mystery_job", "preliminary", when=self.BEFORE)
        self.check("mystery_job", "other", when=self.AFTER)
        self.check("", "other", when=None)

    def test_harvester_suite_inference_uses_classifier(self):
        self.assertEqual(infer_suite_from_job_name("kn_crit_cand"), "Suite 02: Granular Criteria Sweep")
        self.assertEqual(infer_suite_from_job_name("sim_msismall_1M_multi_marathon_array.sh"), "")
        self.assertEqual(infer_suite_from_job_name("kn_telemetry_cron"), "Telemetry Infrastructure")
        self.assertEqual(canonical_suite_name("kn_gw170817_fixed_ensemble"), "Suite 07: GW170817 Rate Limits")


class TestHarvesterFixes(unittest.TestCase):
    """Regression tests for harvester / model / passive-logger bugs."""

    def test_timestamp_placeholders_and_trailing_period(self):
        self.assertIsNone(JobTelemetryRecord._parse_iso("None"))
        self.assertIsNone(JobTelemetryRecord._parse_iso("Unknown"))
        self.assertIsNone(JobTelemetryRecord._parse_iso(""))
        dt = JobTelemetryRecord._parse_iso("Thu Oct 29 16:56:36 CDT 2026.")
        self.assertEqual((dt.month, dt.day, dt.hour), (10, 29, 16))
        self.assertEqual(JobTelemetryRecord._parse_iso("2026-10-20T20:41:25").minute, 41)

    def test_sacct_overrides_seeded_submission_log_record(self):
        seeded = JobTelemetryRecord(job_id="4118182", job_name="kn_criteria_sweep", suite="Suite 02",
                                    submit_time="2026-10-21 20:08:43", req_mem_gb=32.0, req_cpus=1,
                                    data_source="slurm_submission_log")
        sacct = JobTelemetryRecord(job_id="4118182", job_name="run_detection_criteria_sweep.sh",
                                   submit_time="2026-10-19T14:59:04", start_time="2026-10-19T16:35:21",
                                   req_mem_gb=100.0, req_cpus=4, used_walltime_sec=109.0, state="COMPLETED",
                                   data_source="sacct")
        JobHarvester()._merge_records(seeded, sacct)
        self.assertEqual(seeded.job_name, "run_detection_criteria_sweep.sh")
        self.assertEqual(seeded.submit_time, "2026-10-19T14:59:04")
        self.assertEqual(seeded.req_mem_gb, 100.0)
        self.assertEqual(seeded.req_cpus, 4)
        self.assertEqual(seeded.data_source, "sacct")

    def test_newer_sacct_refreshes_running_job(self):
        old = JobTelemetryRecord(job_id="1", state="RUNNING", end_time="Unknown", data_source="sacct")
        new = JobTelemetryRecord(job_id="1", state="COMPLETED", end_time="2026-10-29T10:00:00",
                                 used_walltime_sec=60.0, data_source="sacct")
        JobHarvester()._merge_records(old, new)
        self.assertEqual(old.state, "COMPLETED")
        self.assertEqual(old.end_time, "2026-10-29T10:00:00")

    def test_sacct_cputimeraw_is_not_used_as_cpu_time(self):
        from unittest import mock
        line = ("5000001|kn_crit_cand|msismall|2026-10-29T15:37:36|2026-10-29T15:40:00|2026-10-29T16:40:00|"
                "3600|64G|20G|4|14400|COMPLETED|0:0|n1||03:00:00")
        fake = mock.Mock(stdout="JobIDRaw|...\n" + line + "\n")
        with mock.patch("telemetry.harvester.subprocess.run", return_value=fake):
            recs = JobHarvester().harvest_sacct()
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].used_cpu_sec, 0.0)  # CPUTimeRAW (=allocation) must be ignored
        recs[0].compute_derived_metrics()
        self.assertEqual(recs[0].cpu_efficiency_pct, 0.0)  # unknown, not a spurious 100 %
        self.assertEqual(recs[0].suite, "Suite 02: Granular Criteria Sweep")

    def test_slurm_env_memory_is_megabytes(self):
        self.assertAlmostEqual(slurm_env_mem_gb({"SLURM_MEM_PER_NODE": "131072"}), 128.0, places=3)
        self.assertAlmostEqual(slurm_env_mem_gb({"SLURM_MEM_PER_CPU": "3900"}, cpus=8), 30.469, places=2)
        self.assertEqual(slurm_env_mem_gb({}), 0.0)

    def test_tracker_load_registry_tolerates_blank_cells(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "telemetry").mkdir()
            fields = list(JobTelemetryRecord("").to_dict().keys())
            row = {k: "" for k in fields}
            row.update(job_id="42", job_name="kn_crit_cand", state="COMPLETED", req_cpus="4.0")
            import csv as _csv
            with open(root / "telemetry" / "job_resource_registry.csv", "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow(row)
            recs = JobTelemetryTracker(project_root=root).load_registry()
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].req_cpus, 4)
            self.assertEqual(recs[0].submit_hour, -1)
            self.assertEqual(recs[0].suite, "Suite 02: Granular Criteria Sweep")

    def test_analyzer_queue_stats_ignore_rows_without_timestamps(self):
        recs = [
            JobTelemetryRecord(job_id="1", submit_time="2026-10-20T10:00:00", start_time="2026-10-20T10:30:00",
                               req_mem_gb=16.0),
            JobTelemetryRecord(job_id="2", req_mem_gb=16.0, data_source="reportseff"),  # no timestamps
        ]
        for r in recs:
            r.compute_derived_metrics()
        q = TelemetryAnalyzer(recs).analyze_queue_times()
        self.assertEqual(q["by_memory_tier"]["Standard (<= 32 GB)"]["count"], 1)
        self.assertEqual(q["by_memory_tier"]["Standard (<= 32 GB)"]["median_min"], 30.0)

    def test_analyzer_dedupes_reportseff_array_rows(self):
        recs = [JobTelemetryRecord(job_id="100", job_name="kn_crit_cand", data_source="sacct"),
                JobTelemetryRecord(job_id="101", job_name="kn_crit_cand", data_source="sacct"),
                JobTelemetryRecord(job_id="100", array_task_id="0", data_source="reportseff", cpu_efficiency_pct=40.0),
                JobTelemetryRecord(job_id="100", array_task_id="1", data_source="reportseff", cpu_efficiency_pct=60.0),
                JobTelemetryRecord(job_id="cfg_x", job_name="kn_crit_cand", data_source="script_configuration")]
        a = TelemetryAnalyzer(recs)
        self.assertEqual(len(a.records), 2)
        self.assertEqual(a.analyze_efficiencies()["cpu_efficiency"]["count"], 2)


class TestFairshareSnapshots(unittest.TestCase):
    SAMPLE = """Account|User|RawShares|NormShares|RawUsage|NormUsage|EffectvUsage|FairShare|LevelFS|GrpTRESMins|TRESRunMins
mygroup||1|0.001|123456|0.002|0.002||0.5||
 mygroup|barna314|1|0.25|98765|0.0015|0.40|0.137|0.62||cpu=10
"""

    def test_parse_sshare(self):
        rows = parse_sshare_output(self.SAMPLE, snapshot_time="2026-11-09T00:00:00")
        self.assertEqual(len(rows), 2)
        mine = rows[1]
        self.assertEqual(mine["user"], "barna314")
        self.assertEqual(mine["fairshare"], "0.137")
        self.assertEqual(mine["effective_usage"], "0.40")
        self.assertEqual(mine["snapshot_time"], "2026-11-09T00:00:00")
        self.assertEqual(parse_sshare_output(""), [])
        self.assertEqual(parse_sshare_output("sshare: error: no such user"), [])

    def test_capture_without_sshare_is_noop(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("telemetry.fairshare.shutil.which", return_value=None):
                self.assertIsNone(capture_sshare_snapshot(project_root=Path(tmpdir)))
            self.assertFalse((Path(tmpdir) / "telemetry" / "fairshare_snapshots.csv").exists())

    def test_capture_with_fake_sshare_appends(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = mock.Mock(stdout=self.SAMPLE)
            with mock.patch("telemetry.fairshare.shutil.which", return_value="/usr/bin/sshare"), \
                    mock.patch("telemetry.fairshare.subprocess.run", return_value=fake):
                capture_sshare_snapshot(project_root=Path(tmpdir))
                capture_sshare_snapshot(project_root=Path(tmpdir))
            lines = (Path(tmpdir) / "telemetry" / "fairshare_snapshots.csv").read_text().strip().splitlines()
            self.assertEqual(len(lines), 1 + 2 * 2)  # one header, two snapshots of two rows


def _registry_rows():
    """Small synthetic registry exercising dedupe, propagation and mixed dtypes."""
    base = {k: "" for k in JobTelemetryRecord("").to_dict()}

    def row(**kw):
        r = dict(base)
        r.update({"partition": "msismall", "req_cpus": "1", "state": "COMPLETED", "data_source": "sacct"})
        r.update({k: str(v) for k, v in kw.items()})
        return r

    t_sub, t_start, t_end = "2026-10-28T10:00:00", "2026-10-28T10:10:00", "2026-10-28T11:10:00"
    rows = [
        # Suite 01 array: two sacct task rows (raw ids) + matching reportseff duplicates
        row(job_id=500, job_name="kn_rate_limits_1M_ensemble", submit_time=t_sub, start_time=t_start,
            end_time=t_end, req_cpus=8, req_mem_gb=128, used_walltime_sec=3600, used_max_rss_gb=39,
            used_cpu_sec=28800),
        row(job_id=503, job_name="kn_rate_limits_1M_ensemble", submit_time=t_sub, start_time=t_start,
            end_time="2026-10-28T10:40:00", req_cpus=8, req_mem_gb=128, used_walltime_sec=1800,
            used_max_rss_gb=30, used_cpu_sec=14400),
        row(job_id=500, array_task_id=0, data_source="reportseff", used_walltime_sec=3600, cpu_efficiency_pct=25.0),
        row(job_id=500, array_task_id=1, data_source="reportseff", used_walltime_sec=1800, cpu_efficiency_pct=50.0),
        row(job_id=503, array_task_id=7, data_source="task_log", model="Metzger", n_transients=1000000),
        # OOM job in Suite 02
        row(job_id=600, job_name="kn_criteria_sweep", submit_time=t_sub, start_time=t_start,
            end_time="2026-10-28T10:40:00", req_mem_gb=32, used_walltime_sec=1800, used_max_rss_gb=32,
            state="OUT_OF_MEMORY"),
        # never started (start "None")
        row(job_id=601, job_name="kn_criteria_sweep", submit_time=t_sub, start_time="None",
            end_time=t_sub, req_mem_gb=32, state="CANCELLED by 1"),
        # preliminary + other + pseudo-records
        row(job_id=300, job_name="sim_msismall_1M_marathon.sh", submit_time="2026-07-01T10:00:00",
            start_time="2026-07-01T10:00:00", end_time="2026-07-01T12:00:00", req_mem_gb=250, req_cpus=4,
            used_walltime_sec=7200, used_max_rss_gb=60),
        row(job_id=301, job_name="bash", partition="interactive", submit_time=t_sub, start_time=t_start,
            end_time=t_end, req_mem_gb=16, req_cpus=4, used_walltime_sec=3600),
        row(job_id="cfg_x", job_name="kn_rate_limits_1M_ensemble", data_source="script_configuration"),
        row(job_id="14237735", job_name="sim_msismall_1M.sh", data_source="empirical_benchmark",
            submit_time="2026-10-16 09:30:00"),
        row(job_id="local", data_source="metadata_json"),
    ]
    return rows


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class TestRegistryFrameAndAccounting(unittest.TestCase):
    """telemetry.frame loader + telemetry.accounting fairshare metrics."""

    @classmethod
    def setUpClass(cls):
        import csv as _csv
        import warnings
        from telemetry.frame import load_registry_frame
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "reg.csv"
        rows = _registry_rows()
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a DtypeWarning would fail the test
            cls.df = load_registry_frame(path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_flags_and_dedupe(self):
        df = self.df
        acc = df[df["in_accounting"]]
        self.assertEqual(sorted(acc["job_id"]), ["300", "301", "500", "503", "600", "601"])
        self.assertTrue(df.loc[df["data_source"] == "reportseff", "is_duplicate"].all())
        self.assertTrue(df.loc[df["data_source"] == "task_log", "is_duplicate"].all())
        self.assertFalse(df.loc[df["job_id"] == "cfg_x", "is_job"].item())
        self.assertTrue(df.loc[df["job_id"] == "14237735", "is_seeded"].item())

    def test_name_propagation_classification_and_enrichment(self):
        df = self.df
        rs = df[df["data_source"] == "reportseff"]
        self.assertTrue((rs["job_name_resolved"] == "kn_rate_limits_1M_ensemble").all())
        cat = df[df["in_accounting"]].set_index("job_id")["category"]
        self.assertEqual(cat["500"], "numbered")
        self.assertEqual(cat["300"], "preliminary")
        self.assertEqual(cat["301"], "other")
        self.assertEqual(df.loc[(df["job_id"] == "503") & df["in_accounting"], "model"].item(), "Metzger")

    def test_times_and_cpu_efficiency(self):
        df = self.df[self.df["in_accounting"]].set_index("job_id")
        self.assertTrue(pd.isna(df.loc["601", "queue_wait_min_clean"]))
        self.assertAlmostEqual(df.loc["500", "queue_wait_min_clean"], 10.0)
        # sacct CPU time == allocation (CPUTimeRAW artifact) -> replaced by matched reportseff value
        self.assertAlmostEqual(df.loc["500", "cpu_eff_clean"], 25.0)
        self.assertAlmostEqual(df.loc["503", "cpu_eff_clean"], 50.0)
        self.assertEqual(df.loc["500", "cpu_eff_source"], "reportseff_matched")
        self.assertTrue(pd.isna(df.loc["300", "cpu_eff_clean"]))

    def test_scope_filter(self):
        from telemetry.frame import filter_scope
        self.assertEqual(set(filter_scope(self.df, "numbered")["job_id"]), {"500", "503", "600", "601"})
        self.assertEqual(set(filter_scope(self.df, "preliminary")["job_id"]), {"300"})
        self.assertEqual(len(filter_scope(self.df, "all")), 5)
        self.assertEqual(len(filter_scope(self.df, "all", include_other=True)), 6)
        self.assertEqual(len(filter_scope(self.df, "numbered", date_range=("2030-01-01", None))), 0)
        with self.assertRaises(ValueError):
            filter_scope(self.df, "bogus")

    def test_billing_math(self):
        from telemetry.accounting import add_billing_columns, usage_summary
        from telemetry.frame import filter_scope
        b = add_billing_columns(filter_scope(self.df, "all", include_other=True), mem_per_core_gb=4.0)
        r = b.set_index("job_id").loc["500"]
        self.assertAlmostEqual(r["alloc_core_h"], 8.0)                # 8 cores x 1 h
        self.assertAlmostEqual(r["mem_equiv_core_h"], 32.0)           # 128 GB / 4 GB x 1 h
        self.assertAlmostEqual(r["billed_core_h"], 32.0)              # max(8, 32)
        self.assertTrue(r["mem_dominated"])
        self.assertAlmostEqual(r["needed_core_h"], 39 / 4.0)          # max(2 eff cores, 39/4) x 1 h
        self.assertAlmostEqual(r["slack_core_h"], 32.0 - 39 / 4.0)
        oom = b.set_index("job_id").loc["600"]
        self.assertAlmostEqual(oom["failed_core_h"], oom["billed_core_h"])
        self.assertAlmostEqual(oom["slack_core_h"], 0.0)
        # decomposition adds up
        total = b["useful_core_h"] + b["slack_core_h"] + b["failed_core_h"]
        np.testing.assert_allclose(total.values, b["billed_core_h"].values, atol=1e-9)
        s = usage_summary(b, "category")
        self.assertAlmostEqual(s["share_of_total"].sum(), 1.0)
        self.assertAlmostEqual(s.loc["numbered", "wasted_frac"],
                               (s.loc["numbered", "slack_core_h"] + s.loc["numbered", "failed_core_h"])
                               / s.loc["numbered", "billed_core_h"])
        # empty input does not crash
        self.assertTrue(usage_summary(b.iloc[0:0], "group").empty)

    def test_rightsizing_and_usage_window(self):
        from telemetry.accounting import (add_billing_columns, prior_usage_at, rightsizing_table,
                                          round_up_mem, spearman, usage_rate_grid)
        from telemetry.frame import filter_scope
        b = add_billing_columns(filter_scope(self.df, "numbered"), mem_per_core_gb=4.0)
        rs = rightsizing_table(b, min_completed=2)
        self.assertEqual(len(rs), 1)
        r = rs.iloc[0]
        self.assertEqual(r["rec_mem_gb"], round_up_mem(39 * 1.2))   # 48 GB
        self.assertEqual(r["rec_cpus"], 5)                           # ceil(P95(2,4 eff cores) x 1.25)
        self.assertLess(r["billed_rec_core_h"], r["billed_now_core_h"])
        self.assertEqual(r["action"], "decrease")
        self.assertTrue(rightsizing_table(b, min_completed=10).empty)
        # usage grid conserves billed core-hours
        edges, use = usage_rate_grid(b)
        ran = b.dropna(subset=["start_dt", "end_dt"])
        ran = ran[ran["end_dt"] > ran["start_dt"]]
        self.assertAlmostEqual(use.sum(), ran["billed_core_h"].sum(), places=6)
        pu = prior_usage_at(b, [pd.Timestamp("2026-10-28T09:00"), pd.Timestamp("2026-10-29T12:00")],
                            window_days=7, half_life_days=7)
        self.assertAlmostEqual(pu["window_core_h"].iloc[0], 0.0)
        self.assertAlmostEqual(pu["window_core_h"].iloc[1], ran["billed_core_h"].sum(), places=6)
        self.assertLess(pu["decayed_core_h"].iloc[1], pu["window_core_h"].iloc[1])
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertTrue(np.isnan(spearman([1, 2], [1, 2])))

    def test_real_registry_loads_without_dtype_warning(self):
        import warnings
        from telemetry.frame import load_registry_frame
        path = Path(__file__).resolve().parent.parent / "telemetry" / "job_resource_registry.csv"
        if not path.exists():
            self.skipTest("registry not present")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df = load_registry_frame(path)
        self.assertTrue(set(df["category"]) <= {"numbered", "preliminary", "other"})
        acc = df[df["in_accounting"]]
        self.assertFalse(acc.duplicated(["job_id", "array_task_id"]).any())


class TestNotebookContract(unittest.TestCase):
    NB = Path(__file__).resolve().parent.parent / "notebooks" / "job_resource_usage_analysis.ipynb"

    def setUp(self):
        import json
        self.nb = json.loads(self.NB.read_text(encoding="utf-8"))

    def test_settings_cell_first(self):
        code = [c for c in self.nb["cells"] if c["cell_type"] == "code"]
        src = "".join(code[0]["source"])
        for name in ("ANALYSIS_SCOPE", "INCLUDE_OTHER", "DATE_RANGE", "MEM_PER_CORE_GB"):
            self.assertIn(name, src)

    def test_no_hardcoded_results_in_markdown(self):
        import re
        result_like = re.compile(r"\d{1,3}(,\d{3})+|\d+\.\d+\s*%|~\s*\d|\$\\sim")
        for cell in self.nb["cells"]:
            if cell["cell_type"] == "markdown":
                text = "".join(cell["source"])
                self.assertIsNone(result_like.search(text), f"result-like number in markdown: {text[:120]}")

    def test_builder_check_passes(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_telemetry_notebook", Path(__file__).resolve().parent.parent / "scripts" / "build_telemetry_notebook.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.check_notebook(self.NB), [])

if __name__ == "__main__":
    unittest.main()

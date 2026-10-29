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


if __name__ == "__main__":
    unittest.main()

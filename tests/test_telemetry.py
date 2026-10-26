"""Unit tests for the job telemetry, queue dynamics, and resource rightsizing system."""

import os
from pathlib import Path
import tempfile
import unittest

from telemetry.analyzer import TelemetryAnalyzer
from telemetry.harvester import JobHarvester, parse_memory_to_gb, parse_slurm_time_to_seconds
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

    def test_slurm_time_parsing(self):
        self.assertEqual(parse_slurm_time_to_seconds("01:00:00"), 3600.0)
        self.assertEqual(parse_slurm_time_to_seconds("20:47:00"), 74820.0)
        self.assertEqual(parse_slurm_time_to_seconds("1-02:00:00"), 93600.0)
        self.assertEqual(parse_slurm_time_to_seconds("02:30"), 150.0)
        self.assertEqual(parse_slurm_time_to_seconds("45"), 45.0)

    def test_record_derived_metrics(self):
        rec = JobTelemetryRecord(
            job_id="12345",
            submit_time="2026-10-20 14:00:00",
            start_time="2026-10-20 14:15:00",
            end_time="2026-10-20 15:15:00",
            req_cpus=4,
            req_mem_gb=32.0,
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


if __name__ == "__main__":
    unittest.main()

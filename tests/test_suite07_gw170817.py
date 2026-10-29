#!/usr/bin/env python3
"""Unit tests for Suite 07: Fixed-Parameter GW170817 Kilonova Suite and Suite 01 Comparison."""

import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

sim_mod = importlib.import_module("analysis_suites.07_gw170817_rate_limits.run_gw170817_simulation")
agg_mod = importlib.import_module("analysis_suites.07_gw170817_rate_limits.aggregate_and_compare_suite01")

GW170817_PARAMETERS = sim_mod.GW170817_PARAMETERS
MODEL_ALIASES = sim_mod.MODEL_ALIASES
canonicalize_model_name = sim_mod.canonicalize_model_name
build_gw170817_population = sim_mod.build_gw170817_population
build_gw170817_lightcurve_model = sim_mod.build_gw170817_lightcurve_model

process_suite07_file = agg_mod.process_suite07_file
perform_comparison = agg_mod.perform_comparison
load_suite01_data = agg_mod.load_suite01_data
generate_markdown_report = agg_mod.generate_markdown_report


class TestSuite07GW170817(unittest.TestCase):
    """Test suite for GW170817 fixed-parameter definitions and execution."""

    def test_canonical_models_present(self):
        """Ensure all 4 required canonical archetypes exist in definitions."""
        expected = [
            "GW170817_Bu2026_FixedView",
            "GW170817_Bu2026_Isotropic",
            "GW170817_Metzger_Blue",
            "GW170817_Metzger_Mean",
        ]
        for m in expected:
            self.assertIn(m, GW170817_PARAMETERS)
            cfg = GW170817_PARAMETERS[m]
            self.assertIn("model_family", cfg)
            self.assertIn("description", cfg)
            self.assertIn("citations", cfg)
            self.assertTrue(len(cfg["citations"]) > 0)

    def test_gw170817_bu2026_parameters(self):
        """Verify Bu2026 models match literature-accepted GW170817 values."""
        fixed_cfg = GW170817_PARAMETERS["GW170817_Bu2026_FixedView"]
        self.assertEqual(fixed_cfg["model_family"], "Bu2026")
        self.assertAlmostEqual(fixed_cfg["log10_mej_dyn"], -1.80)
        self.assertAlmostEqual(fixed_cfg["v_ej_dyn"], 0.20)
        self.assertAlmostEqual(fixed_cfg["ye_dyn"], 0.15)
        self.assertAlmostEqual(fixed_cfg["log10_mej_wind"], -1.10)
        self.assertAlmostEqual(fixed_cfg["v_ej_wind"], 0.10)
        self.assertAlmostEqual(fixed_cfg["ye_wind"], 0.35)
        self.assertAlmostEqual(fixed_cfg["inclination_em"], 0.45)
        self.assertFalse(fixed_cfg["vary_inclination"])

        iso_cfg = GW170817_PARAMETERS["GW170817_Bu2026_Isotropic"]
        self.assertEqual(iso_cfg["model_family"], "Bu2026")
        self.assertTrue(iso_cfg["vary_inclination"])
        # Intrinsic ejecta matches fixed view
        self.assertEqual(iso_cfg["log10_mej_dyn"], fixed_cfg["log10_mej_dyn"])
        self.assertEqual(iso_cfg["log10_mej_wind"], fixed_cfg["log10_mej_wind"])

    def test_gw170817_metzger_parameters(self):
        """Verify Metzger models match literature-accepted values."""
        blue_cfg = GW170817_PARAMETERS["GW170817_Metzger_Blue"]
        self.assertEqual(blue_cfg["model_family"], "Metzger")
        self.assertAlmostEqual(blue_cfg["mej"], 0.020)
        self.assertAlmostEqual(blue_cfg["vej"], 0.25)
        self.assertAlmostEqual(blue_cfg["kappa"], 0.50)

        mean_cfg = GW170817_PARAMETERS["GW170817_Metzger_Mean"]
        self.assertEqual(mean_cfg["model_family"], "Metzger")
        self.assertAlmostEqual(mean_cfg["mej"], 0.040)
        self.assertAlmostEqual(mean_cfg["vej"], 0.15)
        self.assertAlmostEqual(mean_cfg["kappa"], 1.00)

    def test_model_name_canonicalization(self):
        """Test alias and case insensitivity handling."""
        self.assertEqual(canonicalize_model_name("GW170817_Bu2026_FixedView"), "GW170817_Bu2026_FixedView")
        self.assertEqual(canonicalize_model_name("bu2026fixed"), "GW170817_Bu2026_FixedView")
        self.assertEqual(canonicalize_model_name("bu2026vary"), "GW170817_Bu2026_Isotropic")
        self.assertEqual(canonicalize_model_name("metzger_blue"), "GW170817_Metzger_Blue")
        self.assertEqual(canonicalize_model_name("metzgermean"), "GW170817_Metzger_Mean")
        with self.assertRaises(ValueError):
            canonicalize_model_name("invalid_model_xyz")

    def test_population_instantiation(self):
        """Test population class instantiation for each model."""
        pop_blue = build_gw170817_population("GW170817_Metzger_Blue", rate=1000.0, z_max=0.15)
        self.assertIsNotNone(pop_blue)
        pop_mean = build_gw170817_population("GW170817_Metzger_Mean", rate=1000.0, z_max=0.15)
        self.assertIsNotNone(pop_mean)
        pop_fixed = build_gw170817_population("GW170817_Bu2026_FixedView", rate=1000.0, z_max=0.15)
        self.assertIsNotNone(pop_fixed)
        pop_iso = build_gw170817_population("GW170817_Bu2026_Isotropic", rate=1000.0, z_max=0.15)
        self.assertIsNotNone(pop_iso)

    def test_lightcurve_model_instantiation(self):
        """Test lightcurve model solver builders."""
        m_metz = build_gw170817_lightcurve_model("GW170817_Metzger_Blue")
        self.assertIsNotNone(m_metz)
        m_bu = build_gw170817_lightcurve_model("GW170817_Bu2026_FixedView")
        self.assertIsNotNone(m_bu)


class TestSuite07AggregationAndComparison(unittest.TestCase):
    """Test suite for aggregation, Poisson rate limits, and Suite 01 comparisons."""

    def test_process_suite07_file_serialization(self):
        """Test parsing serialized JSON and alias serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = Path(tmpdir) / "ztf_GW170817_Metzger_Blue_1000k_task_0_run_0.json"
            sample_data = {
                "n_simulated": 1000000,
                "n_detected": 13175,
                "rate_summaries": [
                    {
                        "transient_type": "Kilonova",
                        "survey_duration_years": 8.0057,
                        "effective_vt_gpc3_yr": 0.11085,
                        "overall_efficiency": 0.013175,
                    }
                ],
            }
            sample_file.write_text(json.dumps(sample_data))

            rows = process_suite07_file(sample_file)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["model"], "GW170817_Metzger_Blue")
            self.assertAlmostEqual(row["effective_vt_gpc3_yr"], 0.11085)
            expected_r95 = -math.log(0.05) / 0.11085
            self.assertAlmostEqual(row["rate_95_upper_limit"], expected_r95, places=3)
            self.assertAlmostEqual(row["rate_95_zero_events"], expected_r95, places=3)

    def test_perform_comparison_math(self):
        """Verify mathematical consistency of comparison calculations."""
        df07 = pd.DataFrame([
            {
                "model": "GW170817_Bu2026_FixedView",
                "effective_vt_gpc3_yr": 0.0180,
                "rate_95_upper_limit": -math.log(0.05) / 0.0180,
                "overall_efficiency": 0.0022,
                "duration_scaled_95": 700.0,
            },
            {
                "model": "GW170817_Bu2026_Isotropic",
                "effective_vt_gpc3_yr": 0.0030,
                "rate_95_upper_limit": -math.log(0.05) / 0.0030,
                "overall_efficiency": 0.00036,
                "duration_scaled_95": 4200.0,
            },
            {
                "model": "GW170817_Metzger_Blue",
                "effective_vt_gpc3_yr": 0.1100,
                "rate_95_upper_limit": -math.log(0.05) / 0.1100,
                "overall_efficiency": 0.0130,
                "duration_scaled_95": 115.0,
            },
        ])

        df01 = pd.DataFrame([
            {
                "model": "Bu2026Fixed",
                "effective_vt_gpc3_yr": 0.0150,
                "rate_95_upper_limit": -math.log(0.05) / 0.0150,
                "overall_efficiency": 0.0018,
            },
            {
                "model": "Bu2026Vary",
                "effective_vt_gpc3_yr": 0.0025,
                "rate_95_upper_limit": -math.log(0.05) / 0.0025,
                "overall_efficiency": 0.0003,
            },
            {
                "model": "Metzger",
                "effective_vt_gpc3_yr": 0.0100,
                "rate_95_upper_limit": -math.log(0.05) / 0.0100,
                "overall_efficiency": 0.0012,
            },
        ])

        metrics = perform_comparison(df07, df01)
        self.assertIn("pairwise_comparisons", metrics)
        self.assertIn("viewing_angle_suppression", metrics)

        # Check viewing angle suppression factor: 0.0180 / 0.0030 = 6.0x
        vas = metrics["viewing_angle_suppression"]
        self.assertAlmostEqual(vas["vt_suppression_factor"], 6.0, places=2)

        # Check pairwise ratios
        pairs = {p["suite07_model"]: p for p in metrics["pairwise_comparisons"]}
        self.assertIn("GW170817_Bu2026_FixedView", pairs)
        p_fix = pairs["GW170817_Bu2026_FixedView"]
        self.assertAlmostEqual(p_fix["vt_ratio_07_to_01"], 0.0180 / 0.0150, places=3)

        self.assertIn("GW170817_Metzger_Blue", pairs)
        p_blue = pairs["GW170817_Metzger_Blue"]
        self.assertAlmostEqual(p_blue["vt_ratio_07_to_01"], 0.1100 / 0.0100, places=3)

    def test_markdown_report_generation(self):
        """Verify markdown report generation produces valid output."""
        metrics = {
            "suite07_summary": {},
            "suite01_summary": {},
            "pairwise_comparisons": [
                {
                    "suite07_model": "GW170817_Bu2026_FixedView",
                    "suite01_model": "Bu2026Fixed",
                    "description": "Fixed polar viewing comparison",
                    "suite07_vt_gpc3_yr": 0.018,
                    "suite01_vt_gpc3_yr": 0.015,
                    "vt_ratio_07_to_01": 1.2,
                    "suite07_r95": 166.4,
                    "suite01_r95": 199.7,
                    "r95_ratio_07_to_01": 0.833,
                    "r95_percent_diff": -16.7,
                }
            ],
            "viewing_angle_suppression": {
                "fixed_vt": 0.018,
                "isotropic_vt": 0.003,
                "vt_suppression_factor": 6.0,
                "fixed_r95": 166.4,
                "isotropic_r95": 998.6,
                "r95_increase_factor": 6.0,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / "REPORT.md"
            generate_markdown_report(metrics, report_file)
            self.assertTrue(report_file.exists())
            text = report_file.read_text()
            self.assertIn("Suite 07: Fixed-Parameter GW170817", text)
            self.assertIn("GW170817_Bu2026_FixedView", text)
            self.assertIn("Viewing Angle Suppression", text)


if __name__ == "__main__":
    unittest.main()

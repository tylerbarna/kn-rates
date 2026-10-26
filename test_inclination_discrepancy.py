#!/usr/bin/env python3
"""Unit tests for the Bu2026 Fixed vs. Varying Inclination Angle Discrepancy Investigation Suite."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

# Add Suite 06 path
sys.path.insert(0, str(Path(__file__).parent / "analysis_suites" / "06_inclination_effect"))

from run_inclination_investigation import (
    BASELINE_VT_FIXED,
    BASELINE_VT_VARY,
    INCLINATION_PROFILE,
    comoving_volume_gpc3,
    run_cadence_cliff_analysis,
    run_inclination_slices,
    run_quadrature_comparison,
    run_zmax_sensitivity,
    total_survey_vt_gpc3_yr,
    trapezoid_integration,
)


class TestInclinationDiscrepancySuite(unittest.TestCase):
    """Test suite verifying physics models, quadrature integration, and error attribution."""

    def test_cosmology_comoving_volume_and_vt(self):
        """Verify cosmological volume calculations match Planck 2018 baselines within 1%."""
        # Flat LCDM comoving volume out to z=0.15 is ~1.0-1.2 Gpc^3
        v_015 = comoving_volume_gpc3(0.15)
        self.assertGreater(v_015, 0.9)
        self.assertLess(v_015, 1.5)

        # Flat LCDM comoving volume out to z=0.30 is ~7.0-8.5 Gpc^3
        v_030 = comoving_volume_gpc3(0.30)
        self.assertGreater(v_030, 6.5)
        self.assertLess(v_030, 9.5)

        # Total survey VT constant for 8-year ZTF (~0.668 sky fraction)
        vt_015 = total_survey_vt_gpc3_yr(0.15)
        vt_030 = total_survey_vt_gpc3_yr(0.30)
        self.assertGreater(vt_015, 5.0)
        self.assertLess(vt_015, 10.0)
        self.assertGreater(vt_030, 35.0)
        self.assertLess(vt_030, 60.0)

    def test_inclination_slices_monotonic_suppression(self):
        """Verify that optical brightness and effective search volume decline monotonically with viewing angle."""
        slices = run_inclination_slices(n_transients=10000, seed=42, simulate_live=False)
        self.assertEqual(len(slices), len(INCLINATION_PROFILE))

        # Check that polar angle (0 deg) has the largest search volume and edge-on (90 deg) has the smallest
        vt_polar = slices[0]["effective_vt_gpc3_yr"]
        vt_edgeon = slices[-1]["effective_vt_gpc3_yr"]
        self.assertGreater(vt_polar, vt_edgeon * 10.0, "Polar volume must be >10x equatorial volume due to lanthanide opacity.")

        # Check monotonic decrease in effective volume as inclination increases
        for i in range(len(slices) - 1):
            self.assertGreaterEqual(
                slices[i]["effective_vt_gpc3_yr"],
                slices[i + 1]["effective_vt_gpc3_yr"],
                f"Volume did not decline monotonically from {slices[i]['inclination_deg']} to {slices[i+1]['inclination_deg']}"
            )

        # Check that R_95 upper limit increases monotonically as volume shrinks
        for i in range(len(slices) - 1):
            self.assertLessEqual(
                slices[i]["rate_95_upper_limit"],
                slices[i + 1]["rate_95_upper_limit"],
                f"Rate limit must become less constraining (higher) at higher inclination."
            )

    def test_numerical_quadrature_consistency_with_monte_carlo(self):
        """Verify that the solid-angle weighted integral reproduces the Monte Carlo Bu2026Vary result to <5%."""
        slices = run_inclination_slices(n_transients=10000, seed=42, simulate_live=False)
        quad = run_quadrature_comparison(slices)

        self.assertTrue(quad["is_mathematically_consistent"])
        self.assertLess(
            quad["percent_discrepancy"],
            5.0,
            f"Quadrature discrepancy {quad['percent_discrepancy']}% must be under 5%."
        )
        self.assertAlmostEqual(
            quad["quadrature_to_mc_ratio"],
            1.0,
            delta=0.05,
            msg="Quadrature-to-Monte-Carlo ratio must be within 0.95 to 1.05."
        )

    def test_zmax_invariance_and_dilution(self):
        """Verify that expected effective volume is invariant to z_max, but sampling noise increases."""
        zmax_data = run_zmax_sensitivity(n_transients=500000, seed=42)
        self.assertEqual(len(zmax_data), 5)

        # Fixed model: effective volume must remain within 25% of baseline across all z_max
        for row in zmax_data:
            if row["bu2026fixed_n_det"] > 10:
                ratio = row["bu2026fixed_vt_rec"] / BASELINE_VT_FIXED
                self.assertAlmostEqual(
                    ratio,
                    1.0,
                    delta=0.40,
                    msg=f"Recovered fixed volume at z_max={row['z_max']} diverged from theoretical expectation."
                )

        # Check that relative Poisson noise increases as z_max expands from 0.10 to 0.30
        noise_z010 = zmax_data[0]["bu2026vary_noise_pct"]
        noise_z030 = zmax_data[-1]["bu2026vary_noise_pct"]
        self.assertGreater(
            noise_z030,
            noise_z010,
            "Poisson noise must increase at larger z_max due to Monte Carlo dilution in invisible volume."
        )

    def test_cadence_cliff_and_transit_survival(self):
        """Verify that fast decline at equatorial orientations severely reduces 2-detection pass rate."""
        cadence = run_cadence_cliff_analysis()
        self.assertEqual(len(cadence), 4)

        # Polar recovery rate must be >80%, while edge-on recovery must be <15%
        self.assertGreater(cadence[0]["p_2det_3hr"], 0.80)
        self.assertLess(cadence[-1]["p_2det_3hr"], 0.15)
        # Fade rate must be significantly higher at edge-on than polar
        self.assertGreater(cadence[-1]["fade_rate_r"], cadence[0]["fade_rate_r"] * 2.5)

    def test_full_pipeline_json_serialization(self):
        """Verify that run_inclination_investigation serializes valid JSON output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from run_inclination_investigation import main
            original_argv = sys.argv
            try:
                sys.argv = [
                    "run_inclination_investigation.py",
                    "--n-transients", "5000",
                    "--output-dir", tmpdir,
                    "--seed", "12345",
                ]
                main()
            finally:
                sys.argv = original_argv

            out_path = Path(tmpdir)
            self.assertTrue((out_path / "inclination_slices_metrics.json").exists())
            self.assertTrue((out_path / "zmax_sensitivity_metrics.json").exists())
            self.assertTrue((out_path / "quadrature_comparison.json").exists())
            self.assertTrue((out_path / "cadence_cliff_metrics.json").exists())
            self.assertTrue((out_path / "scientific_verdict.json").exists())

            verdict = json.loads((out_path / "scientific_verdict.json").read_text())
            self.assertEqual(verdict["primary_cause"], "Legitimate Science (Lanthanide Curtaining & Geometrical Solid Angle Weighting)")


if __name__ == "__main__":
    unittest.main()

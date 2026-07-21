#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_model_outputs import compare_spectra  # noqa: E402


class SpectralComparisonTests(unittest.TestCase):
    def test_identical_signal_has_zero_error(self):
        signal = [((index % 100) - 50) / 50.0 for index in range(4096)]
        metrics = compare_spectra(signal, signal)
        self.assertEqual(metrics["spectral_convergence"], 0.0)
        self.assertEqual(metrics["log_spectral_distance_db"], 0.0)

    def test_changed_signal_has_nonzero_error(self):
        reference = [((index % 100) - 50) / 50.0 for index in range(4096)]
        candidate = [value * 0.9 for value in reference]
        metrics = compare_spectra(reference, candidate)
        self.assertGreater(metrics["spectral_convergence"], 0.0)
        self.assertGreater(metrics["log_spectral_distance_db"], 0.0)


if __name__ == "__main__":
    unittest.main()

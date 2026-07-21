#!/usr/bin/env python3
"""Model-free tests for tools/backend_validation.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "backend_validation.py"
SPEC = importlib.util.spec_from_file_location("backend_validation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


class MetricTests(unittest.TestCase):
    def test_identical_samples_have_perfect_metrics(self) -> None:
        samples = [0.0, 0.25, -0.5, 0.75]
        metrics = validation.compare_samples(samples, samples)
        self.assertEqual(metrics["correlation"], 1.0)
        self.assertEqual(metrics["rms_delta"], 0.0)
        self.assertEqual(metrics["relative_rms_delta"], 0.0)
        self.assertEqual(metrics["max_absolute_delta"], 0.0)
        self.assertEqual(metrics["snr_db"], "inf")

    def test_scaled_samples_have_expected_correlation_and_delta(self) -> None:
        reference = [-0.5, -0.25, 0.25, 0.5]
        candidate = [value * 0.5 for value in reference]
        metrics = validation.compare_samples(reference, candidate)
        self.assertAlmostEqual(metrics["correlation"], 1.0)
        self.assertAlmostEqual(metrics["relative_rms_delta"], 0.5)
        self.assertAlmostEqual(metrics["snr_db"], 20.0 * 0.3010299956639812)

    def test_transcript_normalization_ignores_case_and_punctuation(self) -> None:
        left = validation.normalize_transcript("The Q8 model—works!")
        right = validation.normalize_transcript("the q8 model works")
        self.assertEqual(left, right)

    def test_case_parser_rejects_duplicate_names(self) -> None:
        args = SimpleNamespace(case=["same=one", "same=two"], text=None)
        with self.assertRaises(ValueError):
            validation.parse_cases(args)


class WavTests(unittest.TestCase):
    def test_read_pcm16_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(array("h", [0, 16384, -16384]).tobytes())
            metadata, samples = validation.read_pcm16_wav(path)
        self.assertEqual(metadata["sample_rate"], 24000)
        self.assertEqual(metadata["sample_count"], 3)
        self.assertEqual(samples, [0.0, 0.5, -0.5])


class ReportIntegrationTests(unittest.TestCase):
    def test_fake_cli_produces_machine_readable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            cli = temp / "fake-kokoro-cli"
            cli.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import sys
                    import wave
                    from array import array
                    from pathlib import Path

                    output = Path(sys.argv[sys.argv.index('-o') + 1])
                    backend = sys.argv[sys.argv.index('--backend') + 1]
                    amplitude = 1000 if backend == 'cpu' else 900
                    samples = array('h', [amplitude, -amplitude] * 120)
                    with wave.open(str(output), 'wb') as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(24000)
                        wav.writeframes(samples.tobytes())
                    """
                ),
                encoding="utf-8",
            )
            cli.chmod(0o755)
            model = temp / "model.gguf"
            voice = temp / "voice.gguf"
            model.write_bytes(b"model fixture")
            voice.write_bytes(b"voice fixture")
            output = temp / "results"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--cli", str(cli),
                    "--model", str(model),
                    "--voice", str(voice),
                    "--backend", "vulkan",
                    "--output-dir", str(output),
                    "--runs", "2",
                    "--case", "short=Hello validation",
                    "--source-revision", "test-source",
                    "--ggml-revision", "test-ggml",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["configuration"]["source_revision"], "test-source")
        self.assertEqual(report["configuration"]["ggml_revision"], "test-ggml")
        self.assertTrue(report["cases"][0]["reference"]["deterministic_pcm16"])
        self.assertTrue(report["cases"][0]["candidate"]["deterministic_pcm16"])
        self.assertAlmostEqual(report["cases"][0]["pcm16_comparison"]["correlation"], 1.0)
        self.assertEqual(report["failures"], [])


if __name__ == "__main__":
    unittest.main()

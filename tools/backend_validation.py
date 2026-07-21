#!/usr/bin/env python3
"""Compare fixed-seed Kokoro output across two inference backends.

This tool deliberately uses only the Python standard library plus `curl` for
optional multipart ASR requests. It writes WAVs, process logs, and one JSON
report so experimental results remain inspectable after a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import unicodedata
import wave
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

DEFAULT_TEXT = "The documented Q Eight model works."


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pcm16_wav(path: Path) -> tuple[dict[str, int], list[float]]:
    with wave.open(str(path), "rb") as wav:
        metadata = {
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "sample_rate": wav.getframerate(),
            "sample_count": wav.getnframes(),
        }
        if metadata["channels"] != 1 or metadata["sample_width_bytes"] != 2:
            raise ValueError(
                f"{path}: expected mono PCM16, got {metadata['channels']} channel(s), "
                f"{metadata['sample_width_bytes'] * 8}-bit samples"
            )
        raw = wav.readframes(metadata["sample_count"])

    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return metadata, [value / 32768.0 for value in samples]


def compare_samples(reference: Sequence[float], candidate: Sequence[float]) -> dict[str, Any]:
    count = min(len(reference), len(candidate))
    if count == 0:
        raise ValueError("cannot compare empty waveforms")

    ref = reference[:count]
    cand = candidate[:count]
    delta = [left - right for left, right in zip(ref, cand)]
    ref_mean = math.fsum(ref) / count
    cand_mean = math.fsum(cand) / count
    ref_energy = math.fsum(value * value for value in ref)
    cand_energy = math.fsum(value * value for value in cand)
    delta_energy = math.fsum(value * value for value in delta)
    covariance = math.fsum(
        (left - ref_mean) * (right - cand_mean)
        for left, right in zip(ref, cand)
    )
    ref_variance = math.fsum((value - ref_mean) ** 2 for value in ref)
    cand_variance = math.fsum((value - cand_mean) ** 2 for value in cand)
    denominator = math.sqrt(ref_variance * cand_variance)

    rms_reference = math.sqrt(ref_energy / count)
    rms_candidate = math.sqrt(cand_energy / count)
    rms_delta = math.sqrt(delta_energy / count)
    relative_rms = rms_delta / rms_reference if rms_reference else None
    if delta_energy == 0:
        snr_db: float | str | None = "inf"
    elif ref_energy == 0:
        snr_db = None
    else:
        snr_db = 10.0 * math.log10(ref_energy / delta_energy)

    return {
        "compared_samples": count,
        "rms_reference": rms_reference,
        "rms_candidate": rms_candidate,
        "rms_delta": rms_delta,
        "relative_rms_delta": relative_rms,
        "max_absolute_delta": max(abs(value) for value in delta),
        "correlation": covariance / denominator if denominator else None,
        "snr_db": snr_db,
    }


def normalize_transcript(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.")
    return value or "case"


def parse_cases(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.case and args.text is not None:
        raise ValueError("use either --case or --text, not both")
    if args.case:
        result: list[tuple[str, str]] = []
        seen: set[str] = set()
        for value in args.case:
            if "=" not in value:
                raise ValueError(f"invalid --case {value!r}; expected NAME=TEXT")
            name, text = value.split("=", 1)
            name = safe_name(name)
            if not text or name in seen:
                raise ValueError(f"case names must be unique and case text must not be empty: {value!r}")
            seen.add(name)
            result.append((name, text))
        return result
    return [("baseline", args.text if args.text is not None else DEFAULT_TEXT)]


def run_synthesis(
    args: argparse.Namespace,
    backend: str,
    case_name: str,
    text: str,
    output_dir: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    hashes: list[str] = []

    for run_index in range(args.warmups + args.runs):
        warmup = run_index < args.warmups
        measured_index = run_index - args.warmups + 1
        label = f"warmup-{run_index + 1}" if warmup else f"run-{measured_index}"
        stem = f"{case_name}-{backend}-{label}"
        wav_path = output_dir / f"{stem}.wav"
        log_path = output_dir / f"{stem}.stderr.log"
        command = [
            str(args.cli),
            "-m", str(args.model),
            "-v", str(args.voice),
            "-o", str(wav_path),
            "-l", args.language,
            "-t", str(args.threads),
            "-s", str(args.length_scale),
            "--backend", backend,
            "--device", str(args.device),
        ]
        if args.phonemes:
            command.append("--phonemes")
        command.append(text)

        environment = os.environ.copy()
        environment["KOKORO_SEED"] = str(args.seed)
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            check=False,
        )
        wall_seconds = time.perf_counter() - start
        log_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"{backend} {case_name} {label} failed with exit code "
                f"{completed.returncode}; see {log_path}"
            )
        if not wav_path.is_file():
            raise RuntimeError(f"{backend} {case_name} {label} did not create {wav_path}")

        metadata, _ = read_pcm16_wav(wav_path)
        duration_seconds = metadata["sample_count"] / metadata["sample_rate"]
        wav_hash = file_sha256(wav_path)
        record = {
            "label": label,
            "warmup": warmup,
            "wall_seconds": wall_seconds,
            "audio_seconds": duration_seconds,
            "end_to_end_real_time_factor": wall_seconds / duration_seconds,
            "wav": str(wav_path),
            "wav_sha256": wav_hash,
            "stderr_log": str(log_path),
        }
        records.append(record)
        if not warmup:
            hashes.append(wav_hash)

    measured = [record for record in records if not record["warmup"]]
    timings = [record["wall_seconds"] for record in measured]
    final_wav = Path(measured[-1]["wav"])
    metadata, samples = read_pcm16_wav(final_wav)
    metadata["duration_seconds"] = metadata["sample_count"] / metadata["sample_rate"]
    return {
        "backend": backend,
        "device": args.device,
        "runs": records,
        "timing": {
            "metric": "process wall time: model load + synthesis + WAV write",
            "minimum_seconds": min(timings),
            "median_seconds": statistics.median(timings),
            "maximum_seconds": max(timings),
        },
        "deterministic_pcm16": len(set(hashes)) == 1 if len(hashes) >= 2 else None,
        "final_wav": str(final_wav),
        "wav": metadata,
        "_samples": samples,
    }


def load_reference_wav(path: Path, backend: str, device: int) -> dict[str, Any]:
    metadata, samples = read_pcm16_wav(path)
    metadata["duration_seconds"] = metadata["sample_count"] / metadata["sample_rate"]
    return {
        "backend": backend,
        "device": device,
        "runs": [],
        "timing": None,
        "deterministic_pcm16": None,
        "external_reference": True,
        "final_wav": str(path),
        "wav": metadata,
        "_samples": samples,
    }


def transcribe(args: argparse.Namespace, wav_path: Path) -> dict[str, str]:
    command = [
        args.curl,
        "--fail", "--silent", "--show-error",
        args.transcription_url,
        "-F", f"file=@{wav_path};type=audio/wav",
        "-F", f"model={args.asr_model}",
        "-F", f"language={args.asr_language}",
        "-F", "response_format=json",
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ASR request failed for {wav_path}: {completed.stderr.strip()}")
    try:
        response = json.loads(completed.stdout)
        text = response["text"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"invalid ASR response for {wav_path}: {completed.stdout!r}") from error
    if not isinstance(text, str):
        raise RuntimeError(f"ASR response text is not a string for {wav_path}")
    return {"text": text, "normalized": normalize_transcript(text)}


def public_backend_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True, help="kokoro-cli executable")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--backend", default="vulkan", choices=("cpu", "vulkan", "metal", "cuda"))
    parser.add_argument("--reference-backend", default="cpu", choices=("cpu", "vulkan", "metal", "cuda"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="JSON report path (default: OUTPUT_DIR/report.json)")
    parser.add_argument(
        "--reference-wav",
        type=Path,
        help="reuse a trusted reference WAV instead of running the reference backend",
    )
    parser.add_argument("--text", help="one test utterance")
    parser.add_argument("--case", action="append", metavar="NAME=TEXT", help="repeat for a small corpus")
    parser.add_argument("--phonemes", action="store_true")
    parser.add_argument("--language", default="en-us")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runs", type=int, default=2, help="measured process runs per backend")
    parser.add_argument("--warmups", type=int, default=0, help="unmeasured process runs per backend")
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument("--min-correlation", type=float, help="optional diagnostic regression threshold")
    parser.add_argument("--max-relative-rms", type=float, help="optional diagnostic regression threshold")
    parser.add_argument("--transcription-url", help="OpenAI-compatible /v1/audio/transcriptions URL")
    parser.add_argument("--asr-model", default="nemotron-3.5-asr-streaming-0.6b")
    parser.add_argument("--asr-language", default="en-US")
    parser.add_argument("--curl", default="curl")
    parser.add_argument("--source-revision", help="kokoro.cpp Git revision recorded in the report")
    parser.add_argument("--ggml-revision", help="GGML Git revision recorded in the report")
    args = parser.parse_args(argv)

    try:
        if args.runs < 1 or args.warmups < 0:
            raise ValueError("--runs must be at least 1 and --warmups cannot be negative")
        args.cli = args.cli.resolve(strict=True)
        args.model = args.model.resolve(strict=True)
        args.voice = args.voice.resolve(strict=True)
        if args.reference_wav is not None:
            args.reference_wav = args.reference_wav.resolve(strict=True)
        cases = parse_cases(args)
        if args.reference_wav is not None and len(cases) != 1:
            raise ValueError("--reference-wav can only be used with one text case")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = args.output_dir.resolve()
        report_path = (args.report or args.output_dir / "report.json").resolve()

        report: dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "configuration": {
                "cli": str(args.cli),
                "cli_sha256": file_sha256(args.cli),
                "model": str(args.model),
                "model_sha256": file_sha256(args.model),
                "voice": str(args.voice),
                "voice_sha256": file_sha256(args.voice),
                "reference_backend": args.reference_backend,
                "reference_wav": str(args.reference_wav) if args.reference_wav else None,
                "candidate_backend": args.backend,
                "device": args.device,
                "seed": args.seed,
                "threads": args.threads,
                "length_scale": args.length_scale,
                "language": args.language,
                "phonemes": args.phonemes,
                "runs": args.runs,
                "warmups": args.warmups,
                "platform": platform.platform(),
                "python": sys.version,
                "source_revision": args.source_revision,
                "ggml_revision": args.ggml_revision,
                "relevant_environment": {
                    key: value for key, value in sorted(os.environ.items())
                    if key.startswith(("GGML_", "KOKORO_"))
                },
            },
            "cases": [],
            "failures": [],
        }

        for case_name, text in cases:
            case_dir = args.output_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            reference = (
                load_reference_wav(args.reference_wav, args.reference_backend, args.device)
                if args.reference_wav
                else run_synthesis(args, args.reference_backend, case_name, text, case_dir)
            )
            candidate = run_synthesis(args, args.backend, case_name, text, case_dir)
            comparison = compare_samples(reference["_samples"], candidate["_samples"])
            same_format = all(
                reference["wav"][field] == candidate["wav"][field]
                for field in ("channels", "sample_width_bytes", "sample_rate")
            )
            same_sample_count = reference["wav"]["sample_count"] == candidate["wav"]["sample_count"]
            failures: list[str] = []
            if not same_format:
                failures.append("WAV format differs between reference and candidate")
            if not same_sample_count:
                failures.append("sample count differs between reference and candidate")
            if not args.allow_nondeterministic:
                if reference["deterministic_pcm16"] is False:
                    failures.append("reference output is not fixed-seed deterministic")
                if candidate["deterministic_pcm16"] is False:
                    failures.append("candidate output is not fixed-seed deterministic")
            correlation = comparison["correlation"]
            if args.min_correlation is not None and (
                correlation is None or correlation < args.min_correlation
            ):
                failures.append(
                    f"correlation {correlation!r} is below {args.min_correlation}"
                )
            relative_rms = comparison["relative_rms_delta"]
            if args.max_relative_rms is not None and (
                relative_rms is None or relative_rms > args.max_relative_rms
            ):
                failures.append(
                    f"relative RMS delta {relative_rms!r} exceeds {args.max_relative_rms}"
                )

            asr: dict[str, Any] | None = None
            if args.transcription_url:
                ref_asr = transcribe(args, Path(reference["final_wav"]))
                candidate_asr = transcribe(args, Path(candidate["final_wav"]))
                transcript_match = ref_asr["normalized"] == candidate_asr["normalized"]
                asr = {
                    "reference": ref_asr,
                    "candidate": candidate_asr,
                    "normalized_transcripts_match": transcript_match,
                }
                if not transcript_match:
                    failures.append("normalized ASR transcripts differ")

            case_report = {
                "name": case_name,
                "text": text,
                "reference": public_backend_result(reference),
                "candidate": public_backend_result(candidate),
                "invariants": {
                    "same_wav_format": same_format,
                    "same_sample_count": same_sample_count,
                },
                "pcm16_comparison": comparison,
                "asr": asr,
                "failures": failures,
            }
            report["cases"].append(case_report)
            report["failures"].extend(f"{case_name}: {failure}" for failure in failures)

        report["status"] = "passed" if not report["failures"] else "failed"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        for case in report["cases"]:
            metrics = case["pcm16_comparison"]
            ref_timing = case["reference"]["timing"]
            cand_timing = case["candidate"]["timing"]["median_seconds"]
            ref_timing_text = (
                f"{ref_timing['median_seconds']:.3f}s" if ref_timing else "cached"
            )
            print(
                f"{case['name']}: samples={metrics['compared_samples']} "
                f"correlation={metrics['correlation']:.6f} "
                f"relative_rms={metrics['relative_rms_delta']:.6f} "
                f"wall={ref_timing_text}/{cand_timing:.3f}s "
                f"({args.reference_backend}/{args.backend})"
            )
            if case["asr"]:
                print(
                    f"  ASR match={case['asr']['normalized_transcripts_match']}: "
                    f"{case['asr']['candidate']['text']!r}"
                )
        print(f"report: {report_path}")
        if report["failures"]:
            for failure in report["failures"]:
                print(f"FAIL: {failure}", file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Synthesize identical input with multiple GGUF models and compare WAV output."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from backend_validation import compare_samples, file_sha256, read_pcm16_wav, safe_name


def parse_models(values: list[str]) -> list[tuple[str, Path]]:
    models = []
    names = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --model {value!r}; expected NAME=PATH")
        name, path = value.split("=", 1)
        name = safe_name(name)
        if name in names:
            raise ValueError(f"duplicate model name: {name}")
        model_path = Path(path).resolve(strict=True)
        names.add(name)
        models.append((name, model_path))
    if len(models) < 2:
        raise ValueError("provide at least two --model NAME=PATH arguments")
    return models


def synthesize(args, name: str, model: Path) -> dict:
    wav_path = args.output_dir / f"{name}.wav"
    log_path = args.output_dir / f"{name}.stderr.log"
    command = [
        str(args.cli),
        "-m", str(model),
        "-v", str(args.voice),
        "-o", str(wav_path),
        "-l", args.language,
        "-t", str(args.threads),
        "-s", str(args.length_scale),
    ]
    if args.backend is not None:
        command.extend(["--backend", args.backend])
    if args.device is not None:
        command.extend(["--device", str(args.device)])
    if args.phonemes:
        command.append("--phonemes")
    command.append(args.text)

    environment = os.environ.copy()
    environment["KOKORO_SEED"] = str(args.seed)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"{name} failed with exit code {completed.returncode}; see {log_path}"
        )

    metadata, samples = read_pcm16_wav(wav_path)
    metadata["duration_seconds"] = metadata["sample_count"] / metadata["sample_rate"]
    return {
        "name": name,
        "model": str(model),
        "model_sha256": file_sha256(model),
        "model_size": model.stat().st_size,
        "wav": str(wav_path),
        "wav_sha256": file_sha256(wav_path),
        "stderr_log": str(log_path),
        "wall_seconds": elapsed,
        "audio": metadata,
        "_samples": samples,
    }


def compare_spectra(reference: list[float], candidate: list[float]) -> dict:
    count = min(len(reference), len(candidate))
    frame_size = 1024
    hop_size = 256
    if count < frame_size:
        return {}
    frame_count = 1 + (count - frame_size) // hop_size
    shape = (frame_count, frame_size)
    strides = (hop_size * np.dtype(np.float32).itemsize, np.dtype(np.float32).itemsize)
    window = np.hanning(frame_size).astype(np.float32)
    ref = np.asarray(reference[:count], dtype=np.float32)
    cand = np.asarray(candidate[:count], dtype=np.float32)
    ref_frames = np.lib.stride_tricks.as_strided(ref, shape=shape, strides=strides)
    cand_frames = np.lib.stride_tricks.as_strided(cand, shape=shape, strides=strides)
    ref_mag = np.abs(np.fft.rfft(ref_frames * window, axis=1))
    cand_mag = np.abs(np.fft.rfft(cand_frames * window, axis=1))
    magnitude_delta = cand_mag - ref_mag
    ref_norm = float(np.linalg.norm(ref_mag))
    log_delta_db = 20.0 * (
        np.log10(np.maximum(cand_mag, 1e-7))
        - np.log10(np.maximum(ref_mag, 1e-7))
    )
    return {
        "stft_frame_size": frame_size,
        "stft_hop_size": hop_size,
        "spectral_convergence": (
            float(np.linalg.norm(magnitude_delta)) / ref_norm if ref_norm else None
        ),
        "log_spectral_distance_db": float(
            np.mean(np.sqrt(np.mean(log_delta_db * log_delta_db, axis=1)))
        ),
    }


def audio_comparison(reference: dict, candidate: dict) -> dict:
    metrics = compare_samples(reference["_samples"], candidate["_samples"])
    metrics.update(compare_spectra(reference["_samples"], candidate["_samples"]))
    metrics.update(
        {
            "reference_samples": reference["audio"]["sample_count"],
            "candidate_samples": candidate["audio"]["sample_count"],
            "sample_count_delta": (
                candidate["audio"]["sample_count"]
                - reference["audio"]["sample_count"]
            ),
            "pcm16_identical": reference["wav_sha256"] == candidate["wav_sha256"],
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument(
        "--model", action="append", required=True, metavar="NAME=PATH",
        help="Model variant; repeat at least twice",
    )
    parser.add_argument("--reference", help="Reference model name; defaults to first")
    parser.add_argument("--text", default="The documented Q Eight model works.")
    parser.add_argument("--phonemes", action="store_true")
    parser.add_argument("--language", default="en-us")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", help="Backend passed to CLIs that support --backend")
    parser.add_argument("--device", type=int, help="Device passed to CLIs that support --device")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.cli = args.cli.resolve(strict=True)
    args.voice = args.voice.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = parse_models(args.model)
    reference_name = args.reference or models[0][0]
    if reference_name not in {name for name, _ in models}:
        raise ValueError(f"unknown reference model: {reference_name}")

    variants = [synthesize(args, name, path) for name, path in models]
    reference = next(item for item in variants if item["name"] == reference_name)
    comparisons = {}
    for candidate in variants:
        if candidate["name"] == reference_name:
            continue
        comparisons[candidate["name"]] = audio_comparison(reference, candidate)

    pairwise = {
        f"{left['name']}->{right['name']}": audio_comparison(left, right)
        for left, right in combinations(variants, 2)
    }

    for variant in variants:
        variant.pop("_samples")
    report = {
        "configuration": {
            "reference": reference_name,
            "voice": str(args.voice),
            "voice_sha256": file_sha256(args.voice),
            "text": args.text,
            "phonemes": args.phonemes,
            "language": args.language,
            "threads": args.threads,
            "length_scale": args.length_scale,
            "seed": args.seed,
            "backend": args.backend,
            "device": args.device,
        },
        "variants": variants,
        "comparisons_to_reference": comparisons,
        "pairwise_comparisons": pairwise,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare the dequantized tensors in two GGUF files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFReader, dequantize
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: install models/requirements-convert.txt"
    ) from exc


def tensor_values(tensor) -> np.ndarray:
    return dequantize(tensor.data, tensor.tensor_type).reshape(-1).astype(
        np.float32, copy=False
    )


def compare(reference_path: Path, candidate_path: Path, top: int) -> dict:
    reference = {tensor.name: tensor for tensor in GGUFReader(reference_path).tensors}
    candidate = {tensor.name: tensor for tensor in GGUFReader(candidate_path).tensors}

    missing = sorted(reference.keys() - candidate.keys())
    extra = sorted(candidate.keys() - reference.keys())
    if missing or extra:
        raise ValueError(f"tensor-name mismatch; missing={missing}, extra={extra}")

    total_count = 0
    sum_abs = 0.0
    sum_sq = 0.0
    reference_sq = 0.0
    candidate_sq = 0.0
    cross = 0.0
    global_max = 0.0
    global_max_tensor = ""
    exact_tensors = 0
    type_pairs: dict[str, int] = {}
    per_tensor = []

    for name, ref_tensor in reference.items():
        cand_tensor = candidate[name]
        ref = tensor_values(ref_tensor)
        cand = tensor_values(cand_tensor)
        if ref.shape != cand.shape:
            raise ValueError(
                f"{name}: dequantized shape mismatch {ref.shape} != {cand.shape}"
            )

        diff = cand - ref
        abs_diff = np.abs(diff)
        count = diff.size
        tensor_sum_sq = float(np.sum(diff * diff, dtype=np.float64))
        tensor_ref_sq = float(np.sum(ref * ref, dtype=np.float64))
        tensor_max = float(np.max(abs_diff, initial=0.0))
        tensor_rmse = math.sqrt(tensor_sum_sq / count) if count else 0.0
        tensor_rel_l2 = (
            math.sqrt(tensor_sum_sq / tensor_ref_sq) if tensor_ref_sq else 0.0
        )

        if tensor_max == 0.0:
            exact_tensors += 1
        if tensor_max > global_max:
            global_max = tensor_max
            global_max_tensor = name

        ref_type = ref_tensor.tensor_type.name
        cand_type = cand_tensor.tensor_type.name
        pair = f"{ref_type}->{cand_type}"
        type_pairs[pair] = type_pairs.get(pair, 0) + 1
        per_tensor.append(
            {
                "name": name,
                "reference_type": ref_type,
                "candidate_type": cand_type,
                "elements": count,
                "max_abs": tensor_max,
                "rmse": tensor_rmse,
                "relative_l2": tensor_rel_l2,
            }
        )

        total_count += count
        sum_abs += float(np.sum(abs_diff, dtype=np.float64))
        sum_sq += tensor_sum_sq
        reference_sq += tensor_ref_sq
        candidate_sq += float(np.sum(cand * cand, dtype=np.float64))
        cross += float(np.sum(ref * cand, dtype=np.float64))

    denominator = math.sqrt(reference_sq * candidate_sq)
    per_tensor.sort(key=lambda item: (item["rmse"], item["max_abs"]), reverse=True)
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "reference_size": reference_path.stat().st_size,
        "candidate_size": candidate_path.stat().st_size,
        "size_delta": candidate_path.stat().st_size - reference_path.stat().st_size,
        "tensor_count": len(reference),
        "exact_tensors": exact_tensors,
        "different_tensors": len(reference) - exact_tensors,
        "elements": total_count,
        "type_pairs": dict(sorted(type_pairs.items())),
        "max_abs": global_max,
        "max_abs_tensor": global_max_tensor,
        "rmse": math.sqrt(sum_sq / total_count) if total_count else 0.0,
        "mae": sum_abs / total_count if total_count else 0.0,
        "relative_l2": math.sqrt(sum_sq / reference_sq) if reference_sq else 0.0,
        "cosine_similarity": cross / denominator if denominator else 1.0,
        "top_tensors_by_rmse": per_tensor[:top],
    }


def print_report(report: dict) -> None:
    print(f"reference: {report['reference']} ({report['reference_size']} bytes)")
    print(f"candidate: {report['candidate']} ({report['candidate_size']} bytes)")
    print(f"size delta: {report['size_delta']:+d} bytes")
    print(
        "tensors: "
        f"{report['tensor_count']} total, {report['exact_tensors']} exact, "
        f"{report['different_tensors']} different"
    )
    print(f"elements: {report['elements']}")
    print(f"type pairs: {report['type_pairs']}")
    print(
        f"max abs: {report['max_abs']:.9g} "
        f"({report['max_abs_tensor'] or 'none'})"
    )
    print(f"RMSE: {report['rmse']:.9g}")
    print(f"MAE: {report['mae']:.9g}")
    print(f"relative L2: {report['relative_l2']:.9g}")
    print(f"cosine similarity: {report['cosine_similarity']:.12g}")
    print("top tensors by RMSE:")
    for tensor in report["top_tensors_by_rmse"]:
        print(
            f"  {tensor['name']}: rmse={tensor['rmse']:.9g}, "
            f"max={tensor['max_abs']:.9g}, rel_l2={tensor['relative_l2']:.9g}, "
            f"{tensor['reference_type']}->{tensor['candidate_type']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two GGUF files after dequantizing every tensor"
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    report = compare(args.reference, args.candidate, max(args.top, 0))
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()

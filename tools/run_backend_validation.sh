#!/usr/bin/env bash
# Build one backend and compare fixed-seed CPU/candidate synthesis.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
backend=vulkan
device=0
build_dir=
output_dir=
voice="$root/models/kokoro-voice-af_heart.gguf"
seed=1234
runs=2
warmups=0
threads=4
transcription_url=
full_corpus=0
skip_build=0
quick=0
refresh_reference=0
declare -a models=()

usage() {
    cat <<EOF
Usage: tools/run_backend_validation.sh [options]

Options:
  --backend NAME       vulkan, metal, or cuda (default: vulkan)
  --device N           GPU device index (default: 0)
  --build-dir PATH     CMake build directory
  --output-dir PATH    Validation artifact directory
  --model PATH         Model to test; repeat for multiple models
  --voice PATH         Voice pack
  --seed N             Fixed synthesis seed (default: 1234)
  --threads N          CPU worker threads (default: 4)
  --runs N             Measured process runs per backend (default: 2)
  --warmups N          Unmeasured process runs per backend (default: 0)
  --transcription-url URL
                       OpenAI-compatible transcription endpoint
  --full-corpus        Run short, punctuation, and long fixtures
  --quick              Short, one-run check using a cached CPU reference
  --refresh-reference  Regenerate the cached quick-mode CPU WAV
  --skip-build         Reuse an existing kokoro-cli
  -h, --help           Show this help

The full default tests downloaded F16 and Q8_0 models. Quick mode defaults to
F16 only and, after its first CPU reference generation, runs only the candidate
backend. Use full mode before accepting or committing an optimisation.
EOF
}

while (($#)); do
    case "$1" in
        --backend) backend=${2:?}; shift 2 ;;
        --device) device=${2:?}; shift 2 ;;
        --build-dir) build_dir=${2:?}; shift 2 ;;
        --output-dir) output_dir=${2:?}; shift 2 ;;
        --model) models+=("${2:?}"); shift 2 ;;
        --voice) voice=${2:?}; shift 2 ;;
        --seed) seed=${2:?}; shift 2 ;;
        --threads) threads=${2:?}; shift 2 ;;
        --runs) runs=${2:?}; shift 2 ;;
        --warmups) warmups=${2:?}; shift 2 ;;
        --transcription-url) transcription_url=${2:?}; shift 2 ;;
        --full-corpus) full_corpus=1; shift ;;
        --quick) quick=1; shift ;;
        --refresh-reference) refresh_reference=1; shift ;;
        --skip-build) skip_build=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$backend" in
    vulkan|metal|cuda) ;;
    *) echo "error: --backend must be vulkan, metal, or cuda" >&2; exit 2 ;;
esac

build_dir=${build_dir:-"$root/.validation/build-$backend"}
output_dir=${output_dir:-"$root/.validation/results-$backend"}
if ((${#models[@]} == 0)); then
    models=("$root/models/kokoro-82m-f16.gguf")
    if ((quick == 0)); then
        models+=("$root/models/kokoro-82m-q8_0.gguf")
    fi
fi

for required in "$voice" "${models[@]}"; do
    if [[ ! -f "$required" ]]; then
        echo "error: missing model asset: $required" >&2
        echo "Run models/download-model.sh first or pass --model/--voice." >&2
        exit 2
    fi
done

if ((skip_build == 0)); then
    cmake_args=(
        -S "$root"
        -B "$build_dir"
        -DCMAKE_BUILD_TYPE=Release
        -DKOKORO_BUILD_CLI=ON
        -DKOKORO_BUILD_TESTS=ON
        -DKOKORO_WERROR=ON
        -DKOKORO_VULKAN=OFF
        -DKOKORO_METAL=OFF
        -DKOKORO_CUDA=OFF
    )
    case "$backend" in
        vulkan) cmake_args+=(-DKOKORO_VULKAN=ON) ;;
        metal) cmake_args+=(-DKOKORO_METAL=ON) ;;
        cuda) cmake_args+=(-DKOKORO_CUDA=ON) ;;
    esac
    cmake "${cmake_args[@]}"
    cmake --build "$build_dir" --parallel
    ctest --test-dir "$build_dir" --output-on-failure
fi

cli="$build_dir/kokoro-cli"
if [[ ! -x "$cli" ]]; then
    echo "error: kokoro-cli does not exist or is not executable: $cli" >&2
    exit 2
fi

mkdir -p "$output_dir"
source_revision=$(git -C "$root" rev-parse HEAD 2>/dev/null || printf unknown)
ggml_revision=$(git -C "$root/ggml" rev-parse HEAD 2>/dev/null || printf unknown)
if command -v vulkaninfo >/dev/null 2>&1 && [[ "$backend" == vulkan ]]; then
    vulkaninfo --summary >"$output_dir/vulkaninfo.txt" 2>&1 || true
fi

declare -a cases=(
    --case "baseline=The documented Q Eight model works."
)
if ((quick)); then
    runs=1
    warmups=0
    cases=(--case "quick=Hello.")
fi
if ((full_corpus)); then
    cases+=(
        --case "punctuation=At 10:45, Doctor Smith paid 12 dollars and 50 cents. Did it work?"
        --case "long=GPU inference should preserve speech quality while reducing latency. This longer fixture exercises recurrent state, duration alignment, and waveform reconstruction across several clauses."
    )
fi

for model in "${models[@]}"; do
    name=$(basename "$model")
    name=${name%.gguf}
    result_dir="$output_dir/$name"
    reference_args=()
    if ((quick)); then
        reference_dir="$root/.validation/quick-references"
        mkdir -p "$reference_dir"
        reference_key=$(
            python3 - "$model" "$voice" "$seed" "$threads" <<'PY'
import hashlib
import sys
from pathlib import Path

digest = hashlib.sha256()
for value in sys.argv[1:3]:
    path = Path(value).resolve()
    stat = path.stat()
    digest.update(f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
digest.update("\0".join(sys.argv[3:]).encode())
print(digest.hexdigest()[:16])
PY
        )
        reference_wav="$reference_dir/${name}-${reference_key}.wav"
        if ((refresh_reference)) || [[ ! -f "$reference_wav" ]]; then
            printf 'Generating quick CPU reference: %s\n' "$reference_wav"
            KOKORO_SEED="$seed" "$cli" \
                -m "$model" -v "$voice" -o "$reference_wav" \
                -t "$threads" --backend cpu --device "$device" "Hello."
        fi
        reference_args=(--reference-wav "$reference_wav")
    fi
    command=(
        python3 "$root/tools/backend_validation.py"
        --cli "$cli"
        --model "$model"
        --voice "$voice"
        --backend "$backend"
        --device "$device"
        --output-dir "$result_dir"
        --seed "$seed"
        --threads "$threads"
        --runs "$runs"
        --warmups "$warmups"
        --source-revision "$source_revision"
        --ggml-revision "$ggml_revision"
        "${reference_args[@]}"
        "${cases[@]}"
    )
    if [[ -n "$transcription_url" ]]; then
        command+=(--transcription-url "$transcription_url")
    fi
    printf 'Validating %s on %s device %s\n' "$name" "$backend" "$device"
    "${command[@]}"
done

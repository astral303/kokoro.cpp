#!/bin/sh
# Download the pinned official Kokoro v1.0 weights, then produce the F16
# model, validated Q8_0 layout, and all 54 voice packs.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_dir=${1:-"$script_dir/original/Kokoro-82M"}
output_dir=${2:-"$script_dir/generated"}
set -- $(python3 - "$script_dir/languages.json" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(metadata["source"], metadata["revision"], metadata["checkpoint_sha256"])
PY
)
repo_id=$1
revision=$2
checkpoint_sha256=$3

python3 - "$repo_id" "$revision" "$source_dir" <<'PY'
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    raise SystemExit(
        "Missing conversion dependencies. Run: "
        "python3 -m pip install -r models/requirements-convert.txt"
    )

repo_id, revision, local_dir = sys.argv[1:]
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=local_dir,
    allow_patterns=["config.json", "kokoro-v1_0.pth", "voices/*.pt"],
)
PY

actual_sha256=$(python3 - "$source_dir/kokoro-v1_0.pth" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)
if [ "$actual_sha256" != "$checkpoint_sha256" ]; then
    echo "Checkpoint SHA256 mismatch: expected $checkpoint_sha256, got $actual_sha256" >&2
    exit 1
fi

python3 - "$script_dir/languages.json" "$source_dir/voices" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    voice
    for language in manifest["languages"]
    for voice in language["voices"]
}
actual = {path.stem for path in Path(sys.argv[2]).glob("*.pt")}
if actual != expected:
    missing = ", ".join(sorted(expected - actual)) or "none"
    extra = ", ".join(sorted(actual - expected)) or "none"
    raise SystemExit(f"Official voice set mismatch; missing: {missing}; extra: {extra}")
print(f"Verified {len(actual)} official voice files")
PY

mkdir -p "$output_dir/voices"

python3 "$script_dir/convert-kokoro-to-gguf.py" \
    --input "$source_dir" \
    --checkpoint "$source_dir/kokoro-v1_0.pth" \
    --output "$output_dir/kokoro-82m-f16.gguf" \
    --outtype f16

python3 "$script_dir/convert-kokoro-to-gguf.py" \
    --input "$source_dir" \
    --checkpoint "$source_dir/kokoro-v1_0.pth" \
    --output "$output_dir/kokoro-82m-q8_0.gguf" \
    --outtype q8_0

python3 "$script_dir/convert-kokoro-voices-to-gguf.py" \
    --input-dir "$source_dir/voices" \
    --output-dir "$output_dir/voices"

python3 - "$output_dir" "$repo_id" "$revision" "$checkpoint_sha256" \
    "$script_dir/languages.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
languages = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
voice_names = sorted(
    voice for language in languages["languages"] for voice in language["voices"]
)
files = [
    output_dir / "kokoro-82m-f16.gguf",
    output_dir / "kokoro-82m-q8_0.gguf",
] + [
    output_dir / "voices" / f"kokoro-voice-{voice}.gguf"
    for voice in voice_names
]
missing = [str(path) for path in files if not path.is_file()]
if missing:
    raise SystemExit("missing generated files: " + ", ".join(missing))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "source": sys.argv[2],
    "revision": sys.argv[3],
    "checkpoint_sha256": sys.argv[4],
    "files": [
        {
            "path": str(path.relative_to(output_dir)),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ],
}
(output_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(f"Wrote {output_dir / 'manifest.json'} ({len(files)} GGUF files)")
PY

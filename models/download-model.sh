#!/usr/bin/env bash
# Download Kokoro-82M GGUF model and default voice pack.
# Models are hosted on Hugging Face.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${KOKORO_MODEL_DIR:-$SCRIPT_DIR}"

HF_MODEL="https://huggingface.co/cstr/kokoro-82m-GGUF/resolve/main"
HF_VOICES="https://huggingface.co/cstr/kokoro-voices-GGUF/resolve/main"
MODEL_FILE="kokoro-82m-f16.gguf"
VOICE_FILE="kokoro-voice-af_heart.gguf"

download() {
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        echo "Already exists: $dest"
        return
    fi
    echo "Downloading: $url"
    echo "         to: $dest"
    if command -v curl &>/dev/null; then
        curl --fail --location --progress-bar -o "$dest.part" "$url"
        mv "$dest.part" "$dest"
    elif command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest.part" "$url"
        mv "$dest.part" "$dest"
    else
        echo "Error: need curl or wget" >&2
        exit 1
    fi
}

mkdir -p "$MODEL_DIR"
download "$HF_MODEL/$MODEL_FILE" "$MODEL_DIR/$MODEL_FILE"
download "$HF_VOICES/$VOICE_FILE" "$MODEL_DIR/$VOICE_FILE"

echo ""
echo "Done. Model files in: $MODEL_DIR"
echo ""
echo "Usage:"
echo "  kokoro-cli -m $MODEL_DIR/$MODEL_FILE -v $MODEL_DIR/$VOICE_FILE \"Hello world\""

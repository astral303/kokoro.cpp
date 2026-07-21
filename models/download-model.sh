#!/usr/bin/env bash
# Download Kokoro-82M GGUF model and an official voice pack.
#
# Usage:
#   ./download-model.sh                    # f16 + af_heart (defaults)
#   ./download-model.sh q8_0 ef_dora       # q8_0 + Spanish ef_dora
#   KOKORO_QUANT=q8_0 ./download-model.sh
#   KOKORO_VOICE=ff_siwis ./download-model.sh
#
# Supported quantisations: f16 (default), q8_0
# Models and all 54 voices:
# https://huggingface.co/simonfxr/kokoro.cpp-GGUF
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${KOKORO_MODEL_DIR:-$SCRIPT_DIR}"

QUANT="${1:-${KOKORO_QUANT:-f16}}"
case "$QUANT" in
    f16)   MODEL_FILE="kokoro-82m-f16.gguf" ;;
    q8_0)  MODEL_FILE="kokoro-82m-q8_0.gguf" ;;
    *)     echo "Error: unsupported quantisation '$QUANT' (use f16 or q8_0)" >&2; exit 1 ;;
esac

HF_REPO="https://huggingface.co/simonfxr/kokoro.cpp-GGUF/resolve/main"
VOICE="${2:-${KOKORO_VOICE:-af_heart}}"
VOICE="${VOICE#kokoro-voice-}"
VOICE="${VOICE%.gguf}"
if [[ ! "$VOICE" =~ ^[a-z][fm]_[a-z0-9_]+$ ]]; then
    echo "Error: invalid voice name '$VOICE'" >&2
    exit 1
fi
VOICE_FILE="kokoro-voice-$VOICE.gguf"

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
download "$HF_REPO/$MODEL_FILE" "$MODEL_DIR/$MODEL_FILE"
download "$HF_REPO/voices/$VOICE_FILE" "$MODEL_DIR/$VOICE_FILE"

echo ""
echo "Done. Model files in: $MODEL_DIR"
echo "  Model: $MODEL_FILE ($QUANT)"
echo "  Voice: $VOICE_FILE"
echo ""
echo "Usage:"
echo "  kokoro-cli -m $MODEL_DIR/$MODEL_FILE -v $MODEL_DIR/$VOICE_FILE \"Hello world\""

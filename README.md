# kokoro.cpp

Standalone C/C++ inference for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech, powered by [ggml](https://github.com/ggml-org/ggml).

- **CPU-only** — no GPU required, runs anywhere
- **C ABI** — suitable for cgo, FFI, or direct embedding
- **Minimal dependencies** — pinned ggml submodule + espeak-ng (system)
- **24 kHz output** — mono float32 PCM or 16-bit WAV via CLI

## Quick Start

### 1. Install espeak-ng

```bash
# Debian/Ubuntu
sudo apt install libespeak-ng-dev

# Fedora
sudo dnf install espeak-ng-devel

# macOS
brew install espeak-ng

# Nix
nix-shell -p espeak-ng
```

### 2. Build

```bash
git submodule update --init
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### 3. Download model

```bash
./models/download-model.sh
```

### 4. Synthesize

```bash
./build/kokoro-cli \
  -m models/kokoro-82m-f16.gguf \
  -v models/kokoro-voice-af_heart.gguf \
  "Hello, this is kokoro dot cpp speaking!"
```

Output: `output.wav` (24 kHz, 16-bit mono).

## API

```c
#include "kokoro.h"

// Initialize
struct kokoro_context_params params = kokoro_context_default_params();
struct kokoro_context *ctx = kokoro_init_from_file("model.gguf", params);
kokoro_load_voice_pack(ctx, "voice.gguf");

// Synthesize
int n_samples = 0;
float *pcm = kokoro_synthesize(ctx, "Hello world", &n_samples);

// Use PCM (24 kHz mono float32)...

kokoro_pcm_free(pcm);
kokoro_free(ctx);
```

### Pre-phonemized input

Skip espeak-ng entirely by passing IPA directly:

```c
float *pcm = kokoro_synthesize_phonemes(ctx, "həlˈoʊ wˈɜːld", &n_samples);
```

## Architecture

Kokoro-82M uses the StyleTTS2 / iSTFTNet architecture:

```
Text → espeak-ng → IPA phonemes → token IDs
  → PL-BERT (12-layer shared ALBERT)
  → TextEncoder (Conv1d + bidir LSTM)
  → ProsodyPredictor (durations, F0, energy)
  → Decoder (AdainResBlk1d stack)
  → Generator (iSTFTNet with Snake-α resblocks)
  → iSTFT → 24 kHz audio
```

## License

The kokoro.cpp source code is MIT-licensed (see [LICENSE](LICENSE)).

**Important**: This project requires [espeak-ng](https://github.com/espeak-ng/espeak-ng) as a directly-linked dependency. espeak-ng is licensed under GPLv3+. When you build and distribute the combined binary (library + espeak-ng), the result is subject to GPLv3+ terms. See [NOTICE](NOTICE) for full details.

The current build intentionally treats espeak-ng as a hard dependency. Even
applications that call `kokoro_synthesize_phonemes()` must link it.

## Model Downloads

Models are **not** included in this repository. Download them:

```bash
./models/download-model.sh
```

Or manually from: https://huggingface.co/cstr/kokoro-82m-GGUF

## Provenance

Inference code derived from [CrispASR](https://github.com/CrispStrobe/CrispASR). See [NOTICE](NOTICE) for full provenance and third-party information.

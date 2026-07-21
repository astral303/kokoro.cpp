# kokoro.cpp

Standalone C/C++ inference for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech, powered by [ggml](https://github.com/ggml-org/ggml).

- **CPU and GPU** — CPU fallback plus Vulkan, Metal, and CUDA backends
- **C ABI** — suitable for cgo, FFI, Python ctypes, Rust bindgen
- **24 kHz mono** — non-autoregressive, full-utterance synthesis
- **Minimal CPU dependencies** — pinned ggml submodule + espeak-ng (system)

## Quick Start

```bash
# Install espeak-ng (Debian/Ubuntu)
sudo apt install libespeak-ng-dev

# After cloning kokoro.cpp
git submodule update --init

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
# Add -DKOKORO_VULKAN=OFF for a CPU-only Linux build.
cmake --build . -j$(nproc)

# Download model (f16 default; pass q8_0 for quantised)
../models/download-model.sh

# Synthesize
./kokoro-cli -m ../models/kokoro-82m-f16.gguf \
             -v ../models/kokoro-voice-af_heart.gguf \
             "Hello, this is kokoro dot cpp!"
```

## API

### Structured output (recommended)

```c
#include "kokoro.h"

struct kokoro_context_params params = kokoro_context_default_params();
params.backend = KOKORO_BACKEND_AUTO; // or CPU, VULKAN, METAL, CUDA
params.device = 0;
struct kokoro_context *ctx = kokoro_init_from_file("model.gguf", params);
kokoro_load_voice_pack(ctx, "voice.gguf");

struct kokoro_audio audio = {0};
enum kokoro_status rc = kokoro_synthesize_audio(ctx, "Hello world", &audio);
if (rc == KOKORO_STATUS_OK) {
    // audio.samples: float*, audio.n_samples, audio.sample_rate (24000)
    write_wav("out.wav", audio.samples, audio.n_samples, audio.sample_rate);
}
kokoro_audio_free(&audio);
kokoro_free(ctx);
```

### Raw pointer (compatible with older code)

```c
int n = 0;
float *pcm = kokoro_synthesize(ctx, "Hello", &n);
// Use pcm[0..n-1] at 24 kHz...
kokoro_pcm_free(pcm);
```

### Pre-phonemised input

```c
kokoro_synthesize_phonemes_audio(ctx, "həlˈoʊ wˈɜːld", &audio);
```

### Error handling

```c
if (kokoro_synthesize_audio(ctx, text, &audio) != KOKORO_STATUS_OK) {
    fprintf(stderr, "Error: %s\n", kokoro_last_error());
}
```

### Custom logging

```c
void my_log(enum kokoro_log_level level, const char *msg, void *ud) {
    if (level >= KOKORO_LOG_WARN) printf("[kokoro] %s\n", msg);
}
kokoro_log_set(my_log, NULL);
```

kokoro.cpp logging is disabled by default and the library never writes directly
to stdout or stderr. `kokoro_log_set()` controls only kokoro.cpp diagnostics.
It deliberately leaves GGML's process-global logger untouched; applications
embedding a shared GGML library must configure GGML logging themselves.

## API Design Notes

| Feature | API | Notes |
|---------|-----|-------|
| Visibility | `KOKORO_API` macro | Static builds propagate `KOKORO_STATIC` |
| ABI check | `KOKORO_ABI_VERSION` | First field in params; exact version required |
| Build identity | `kokoro_version()` | Returns "kokoro.cpp 0.1.0 (hash)" |
| Errors | `kokoro_last_error()` | Thread-local, errno-style |
| Logging | `kokoro_log_set(cb, ud)` | Process-wide, thread-safe install |
| Backend | `kokoro_get_backend/name` | Reports the backend selected for a context |
| Audio output | `struct kokoro_audio` | POD; freed with `kokoro_audio_free` |
| Voice query | `kokoro_voice_name/max_phonemes` | After voice pack load |
| Thread safety | Context is NOT thread-safe | One context per thread, or serialise |

No streaming or cancellation API is provided yet. Kokoro computes a complete
utterance before waveform reconstruction, so it has no autoregressive token
loop from which to emit useful incremental audio. Callers can obtain
sentence-level streaming by splitting text and synthesizing each sentence.

## Model Compatibility

The runtime consumes the GGUF files published by CrispASR's `cstr` account.
Model and voice files are separate and both are required.

### Verified in this repository

These combinations were downloaded, loaded on the CPU backend, and used to
produce valid 24 kHz WAV files:

| Model file | Size | Voice file | Result |
|------------|------|------------|--------|
| [`kokoro-82m-f16.gguf`](https://huggingface.co/cstr/kokoro-82m-GGUF/resolve/main/kokoro-82m-f16.gguf) | 163,728,096 bytes (156 MiB) | [`kokoro-voice-af_heart.gguf`](https://huggingface.co/cstr/kokoro-voices-GGUF/resolve/main/kokoro-voice-af_heart.gguf) | Verified |
| [`kokoro-82m-q8_0.gguf`](https://huggingface.co/cstr/kokoro-82m-GGUF/resolve/main/kokoro-82m-q8_0.gguf) | 141,322,336 bytes (135 MiB) | `kokoro-voice-af_heart.gguf` | Verified |

Download either verified model with:

```bash
./models/download-model.sh f16
./models/download-model.sh q8_0
# Equivalent: KOKORO_QUANT=q8_0 ./models/download-model.sh
```

`f16` is the quality/reference default. `q8_0` is smaller and has been
validated for loading and synthesis, but this project has not yet published
an objective quality comparison between them.

Both models have also been synthesized through Vulkan on an AMD Radeon RX 7900
XTX. Fixed-seed CPU and Vulkan output had identical duration and transcription
but was not PCM-identical, as expected from backend-specific floating-point
execution.

### Published voice packs

The [`cstr/kokoro-voices-GGUF`](https://huggingface.co/cstr/kokoro-voices-GGUF)
repository currently publishes:

| File | Intended language | Validation here |
|------|-------------------|-----------------|
| `kokoro-voice-af_heart.gguf` | American English | Verified with F16 and Q8_0 |
| `kokoro-voice-ef_dora.gguf` | Spanish | Format-compatible; not tested here |
| `kokoro-voice-ff_siwis.gguf` | French | Format-compatible; not tested here |
| `kokoro-voice-df_eva.gguf` | German | Format-compatible; not tested here |
| `kokoro-voice-df_victoria.gguf` | German | Format-compatible; not tested here |
| `kokoro-voice-dm_bernd.gguf` | German | Format-compatible; not tested here |
| `kokoro-voice-dm_martin.gguf` | German | Format-compatible; not tested here |

“Format-compatible” means the file uses the required `kokoro-voice`
architecture and `voice.pack` layout; it is not a claim that this project has
quality-tested the voice.

### German backbone (not yet validated here)

[`cstr/kokoro-de-hui-base-GGUF`](https://huggingface.co/cstr/kokoro-de-hui-base-GGUF)
publishes:

- `kokoro-de-hui-base-f16.gguf`
- `kokoro-de-hui-base-q8_0.gguf`

They use the same GGUF architecture and are expected to load, but neither has
been exercised by this standalone project yet. Pair them with a German `df_*`
or `dm_*` voice and select an espeak-ng German voice such as `-l de`.

### Constraints

- **Max input**: 510 raw phoneme tokens. PL-BERT has 512 positions and the
  runtime adds two boundary tokens. Split longer text before synthesis.
- **Phonemizer**: the language passed with `kokoro_set_language()` or CLI
  `-l` is an espeak-ng voice name such as `en-us`, `es`, `fr`, or `de`.
- **CJK caveats**: Mandarin tone numbers are stripped because the 178-token
  vocabulary has no tone slots. espeak-ng's Japanese kanji coverage is
  incomplete. The runtime logs warnings for these cases.
- **Output**: model-declared sample rate, currently 24 kHz; mono float32 PCM.
- **Supported published weight types**: F16 and Q8_0 files listed above.
  Other quantizations are unvalidated and unsupported.
- **Not directly supported**: raw PyTorch `.pth`/`.pt` checkpoints. Convert
  them with CrispASR's `convert-kokoro-to-gguf.py` and
  `convert-kokoro-voice-to-gguf.py`, preserving the expected metadata and
  tensor names.

### GGUF format

Models must declare `general.architecture = "kokoro"` and contain the
CrispASR Kokoro tensor names plus `tokenizer.ggml.tokens`. Voice packs must
declare `general.architecture = "kokoro-voice"` and contain an F32
`voice.pack` tensor. GGUF is only the container format: llama.cpp and
whisper.cpp do not implement this Kokoro architecture.

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
ctest  # run model-free tests
```

### Options

| CMake Option | Default | Description |
|--------------|---------|-------------|
| `KOKORO_BUILD_CLI` | ON standalone, OFF as subproject | Build kokoro-cli tool |
| `KOKORO_BUILD_TESTS` | ON standalone, OFF as subproject | Build and register tests |
| `KOKORO_STATIC` | ON | Static library (propagates KOKORO_STATIC) |
| `KOKORO_WERROR` | OFF | Treat warnings in kokoro.cpp sources as errors |
| `KOKORO_VULKAN` | ON except Apple | Build the Vulkan backend |
| `KOKORO_METAL` | ON on Apple | Build the Metal backend |
| `KOKORO_CUDA` | OFF | Build the CUDA backend |
| `KOKORO_GGML_DIR` | bundled `ggml/` | Use a compatible external ggml source checkout |

At runtime, `KOKORO_BACKEND_AUTO` tries Metal, CUDA, Vulkan, then CPU among the
compiled backends. An explicitly requested unavailable backend is an error.
The CLI exposes the same selection through `--backend` and `--device`.
Vulkan builds require the Vulkan loader, headers, and a shader compiler
(`glslc` or `glslangValidator`); CUDA builds require the CUDA toolkit.

GPU optimisation goals, known graph hazards, current numerical baselines, and
the quick/full CPU-to-GPU validation workflow are documented in
[GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md).

### Embedding and external ggml

When kokoro.cpp is added to another CMake project, CLI and test targets are
disabled by default. Link the namespaced alias:

```cmake
add_subdirectory(path/to/kokoro.cpp)
target_link_libraries(my_app PRIVATE kokoro::kokoro)
```

To build against a separate ggml checkout instead of the bundled submodule:

```bash
cmake -S . -B build -DKOKORO_GGML_DIR=/path/to/ggml
```

If the parent project has already created a `ggml` target, kokoro.cpp reuses it
instead of calling `add_subdirectory()` again. Its source directory is normally
inferred from the target. Set `KOKORO_GGML_DIR` explicitly if it cannot be
inferred.

The bundled dependency is the official upstream ggml project. An external
checkout must expose the operations and private backend header used by
kokoro.cpp, so arbitrary older or newer releases are not guaranteed to work.
The bundled submodule remains the compatibility baseline and source of truth
for the tested revision.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `KOKORO_ESPEAK_DATA_PATH` | Override espeak-ng data directory |
| `KOKORO_BENCH=1` | Print per-stage timing to log. Also emits GPU lifecycle subphases (`stage.graph_build` / `.reset_alloc` / `.upload` / `.compute` / `.download` / `.host`) and CPU harmonic subphases (`har_phase.*`). Harmonic phase times are collected on the worker and logged only after join on the caller thread. |
| `KOKORO_BENCH_ITERS=N` | CLI only: repeat synthesis N times in-process (requires `KOKORO_BENCH=1`; last WAV is written) |
| `KOKORO_SEED=N` | Fix RNG seed for reproducible output |
| `KOKORO_FASTCONV=0` | Disable FASTCONV kernel optimisation |
| `KOKORO_HAR_OVERLAP=0` | Disable Vulkan harmonic/GPU overlap for A/B testing |
| `KOKORO_LSTM_BATCH=0` | Disable Vulkan bidirectional-LSTM batching for A/B testing |
| `KOKORO_DEBUG_DURATIONS=1` | Log raw and rounded token durations via the Kokoro callback |
| `KOKORO_GGUF_MMAP=0` | Disable mmap for model loading |
| `KOKORO_MLOCK=1` | Lock model pages in memory |

GGML backend environment variables are not Kokoro options. In particular,
`GGML_VK_ALLOW_GRAPHICS_QUEUE` is a runtime, presence-based GGML Vulkan control
(`=0` still enables it). A standalone host may set it before the first GGML
Vulkan registration/device query to opt into the measured RADV queue path.
Kokoro never sets it because an embedding process may share one external GGML
instance with other models. The linked external GGML revision must implement
the variable; older/forked implementations may silently ignore it. See
[GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md#radv-graphics-capable-queue-selection--confirmed-improvement)
for compatibility, timing, and benchmark details.

## License

The kokoro.cpp source code is MIT-licensed (see [LICENSE](LICENSE)).

**Important**: This project requires [espeak-ng](https://github.com/espeak-ng/espeak-ng) as a directly-linked dependency. espeak-ng is licensed under GPLv3+. When you build and distribute the combined binary, the result is subject to GPLv3+ terms. See [NOTICE](NOTICE) for full details.

The current build intentionally treats espeak-ng as a hard dependency, even
when callers provide precomputed phonemes. A non-GPL build would require a
source-level build option that removes this link; no such mode is supported
today.

## Provenance

Inference code derived from [CrispASR](https://github.com/CrispStrobe/CrispASR). See [NOTICE](NOTICE) for full attribution and third-party information.

#pragma once

// kokoro.h — public C ABI for Kokoro TTS inference (kokoro.cpp).
//
// Pure C99. Consumable from C, C++, cgo, Python ctypes, Rust bindgen.
// Follows whisper.h / qwentts.h conventions: extern "C" linkage,
// POD structs, const char * UTF-8 strings, status enums.
//
// Architecture: hexgrad/Kokoro-82M (StyleTTS2 / iSTFTNet).
// Output: 24 kHz mono float32 PCM, non-autoregressive (full utterance).
//
// Thread safety:
//   - kokoro_version(), kokoro_last_error(): safe from any thread.
//   - kokoro_log_set(): process-wide, thread-safe install. Log delivery does
//     not hold the install mutex, so callbacks may replace themselves.
//   - kokoro_context: NOT thread-safe. Do not call synthesize/load
//     concurrently on the same context. Create one context per thread
//     or serialise access externally.
//   - Multiple contexts on different threads are independent and safe.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Symbol visibility
// ---------------------------------------------------------------------------

#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(KOKORO_STATIC)
#        define KOKORO_API
#    elif defined(KOKORO_BUILD)
#        define KOKORO_API __declspec(dllexport)
#    else
#        define KOKORO_API __declspec(dllimport)
#    endif
#elif defined(__GNUC__) || defined(__clang__)
#    define KOKORO_API __attribute__((visibility("default")))
#else
#    define KOKORO_API
#endif

// ---------------------------------------------------------------------------
// ABI version
// ---------------------------------------------------------------------------

// Incremented when public POD layouts change. Callers should use
// kokoro_context_default_params(), which sets this field. The library rejects
// parameter structs whose version does not exactly match.
#define KOKORO_ABI_VERSION 2

// ---------------------------------------------------------------------------
// Version / build identity
// ---------------------------------------------------------------------------

// Returns a static string "kokoro.cpp <version> (<git-hash>)" identifying
// the build. Safe from any thread, no allocation. Valid for process lifetime.
KOKORO_API const char * kokoro_version(void);

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

enum kokoro_status {
    KOKORO_STATUS_OK             =  0,
    KOKORO_STATUS_INVALID_PARAMS = -1,
    KOKORO_STATUS_LOAD_FAILED    = -2,
    KOKORO_STATUS_NO_VOICE       = -3,
    KOKORO_STATUS_SYNTH_FAILED   = -4,
};

// Returns the last error message for the calling thread (NUL terminated
// UTF-8). errno-style: meaningful immediately after a failure; stale after
// any subsequent kokoro_* call on the same thread. Thread-local storage.
KOKORO_API const char * kokoro_last_error(void);

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

enum kokoro_log_level {
    KOKORO_LOG_DEBUG = 0,
    KOKORO_LOG_INFO  = 1,
    KOKORO_LOG_WARN  = 2,
    KOKORO_LOG_ERROR = 3,
};

// Callback signature. msg has no trailing newline. Must be reentrant.
typedef void (*kokoro_log_cb)(enum kokoro_log_level level, const char * msg, void * user_data);

// Install a process-wide callback for kokoro.cpp diagnostics. NULL disables
// kokoro.cpp logging. This does not modify GGML's process-global logger;
// embedders that share GGML must configure GGML logging themselves.
// Thread-safe install; takes effect on subsequent log delivery. The caller
// must keep cb and user_data valid while synthesis may still be running.
KOKORO_API void kokoro_log_set(kokoro_log_cb cb, void * user_data);

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct kokoro_context;

enum kokoro_backend {
    KOKORO_BACKEND_AUTO    = 0, // Metal, CUDA, or Vulkan if available; otherwise CPU
    KOKORO_BACKEND_CPU     = 1,
    KOKORO_BACKEND_VULKAN  = 2,
    KOKORO_BACKEND_METAL   = 3,
    KOKORO_BACKEND_CUDA    = 4,
};

struct kokoro_context_params {
    int   abi_version;      // must be KOKORO_ABI_VERSION
    int   n_threads;
    int   verbosity;        // 0=silent, 1=normal, 2=verbose
    float length_scale;     // duration multiplier: 1.0=default, >1=slower, <1=faster
    char  espeak_lang[32];  // espeak-ng voice, default "en-us"
    enum kokoro_backend backend;
    int   device;           // zero-based GPU device; ignored by CPU and Metal
};

KOKORO_API struct kokoro_context_params kokoro_context_default_params(void);

// Load Kokoro GGUF model. Returns NULL on failure (check kokoro_last_error).
KOKORO_API struct kokoro_context * kokoro_init_from_file(const char * path_model,
                                                        struct kokoro_context_params params);

KOKORO_API void kokoro_free(struct kokoro_context * ctx);

KOKORO_API void kokoro_set_n_threads(struct kokoro_context * ctx, int n_threads);

// Actual selected backend (AUTO is never returned for a valid context).
KOKORO_API enum kokoro_backend kokoro_get_backend(const struct kokoro_context * ctx);

// Human-readable selected backend name, or NULL for a null context.
KOKORO_API const char * kokoro_backend_name(const struct kokoro_context * ctx);

// Runtime length-scale. Clamped to [0.25, 4.0].
KOKORO_API void kokoro_set_length_scale(struct kokoro_context * ctx, float scale);

// Load a voice-pack GGUF. Returns KOKORO_STATUS_OK (0) on success.
KOKORO_API enum kokoro_status kokoro_load_voice_pack(struct kokoro_context * ctx, const char * path);

// Override espeak-ng language. Returns KOKORO_STATUS_OK on success.
KOKORO_API enum kokoro_status kokoro_set_language(struct kokoro_context * ctx, const char * espeak_lang);

// Drop internal phoneme cache entries.
KOKORO_API void kokoro_phoneme_cache_clear(struct kokoro_context * ctx);

// ---------------------------------------------------------------------------
// Voice query
// ---------------------------------------------------------------------------

// Returns the loaded voice pack name, or NULL if none loaded. The pointer
// remains valid until the next voice-pack load or kokoro_free().
KOKORO_API const char * kokoro_voice_name(const struct kokoro_context * ctx);

// Returns the max phoneme count the loaded voice supports, or 0.
KOKORO_API int kokoro_voice_max_phonemes(const struct kokoro_context * ctx);

// ---------------------------------------------------------------------------
// Synthesis — raw pointer API (original, preserved for compatibility)
// ---------------------------------------------------------------------------

// Tokenise IPA phonemes → model vocab IDs. Caller frees with free().
KOKORO_API int32_t * kokoro_phonemes_to_ids(struct kokoro_context * ctx,
                                            const char * phonemes, int * out_n);

// Text → 24 kHz mono float32 PCM (runs espeak-ng phonemizer).
// Caller frees with kokoro_pcm_free.
KOKORO_API float * kokoro_synthesize(struct kokoro_context * ctx,
                                     const char * text, int * out_n_samples);

// IPA → 24 kHz mono float32 PCM (skips phonemizer).
// Caller frees with kokoro_pcm_free.
KOKORO_API float * kokoro_synthesize_phonemes(struct kokoro_context * ctx,
                                              const char * phonemes, int * out_n_samples);

KOKORO_API int kokoro_sample_rate(const struct kokoro_context * ctx);

KOKORO_API void kokoro_pcm_free(float * pcm);

// ---------------------------------------------------------------------------
// Synthesis — structured output API
// ---------------------------------------------------------------------------

// Structured audio output. Zero-initialise before first use.
struct kokoro_audio {
    float * samples;       // mono PCM, malloc'd by library
    int     n_samples;     // length in samples
    int     sample_rate;   // 24000
    int     channels;      // 1 (mono)
};

// Release samples and reset struct. Safe on zero-initialised struct.
KOKORO_API void kokoro_audio_free(struct kokoro_audio * audio);

// Synthesise text with structured output. Fills/resets *out.
// Returns KOKORO_STATUS_OK on success.
KOKORO_API enum kokoro_status kokoro_synthesize_audio(struct kokoro_context * ctx,
                                                      const char * text,
                                                      struct kokoro_audio * out);

// Synthesise from IPA phonemes with structured output.
KOKORO_API enum kokoro_status kokoro_synthesize_phonemes_audio(struct kokoro_context * ctx,
                                                               const char * phonemes,
                                                               struct kokoro_audio * out);

#ifdef __cplusplus
}
#endif

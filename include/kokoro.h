#pragma once

// kokoro.h — public C ABI for Kokoro TTS inference.
//
// Implements hexgrad/Kokoro-82M (StyleTTS2 / iSTFTNet architecture).
// Pre-trained voices ship as separate per-voice GGUFs (arch =
// "kokoro-voice") containing one F32 tensor `voice.pack[max_phon, 1,
// 256]`. Index by phoneme length L: ref_s = voice.pack[L-1, 0, :],
// split as [predictor_style 0:128 | decoder_style 128:256].
//
// Phonemizer: espeak-ng is linked directly and called in-process via
// espeak_TextToPhonemes(). Set KOKORO_ESPEAK_DATA_PATH to override the
// espeak-ng-data location. Pass already-IPA strings via
// kokoro_synthesize_phonemes() to bypass the phonemizer entirely.
// The vocab (178 IPA symbols) lives in the GGUF as
// `tokenizer.ggml.tokens`.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kokoro_context;

struct kokoro_context_params {
    int n_threads;
    int verbosity;        // 0=silent, 1=normal, 2=verbose
    float length_scale;   // per-phoneme duration multiplier.
                          // 1.0 = default; >1.0 = slower; <1.0 = faster.
                          // Applied BEFORE banker's-round + clamp-min-1.
                          // Read on every synthesize call.
    char espeak_lang[32]; // espeak-ng -v LANG, default "en-us"
};

struct kokoro_context_params kokoro_context_default_params(void);

// Load Kokoro GGUF model. Returns nullptr on failure.
struct kokoro_context* kokoro_init_from_file(const char* path_model, struct kokoro_context_params params);

void kokoro_free(struct kokoro_context* ctx);
void kokoro_set_n_threads(struct kokoro_context* ctx, int n_threads);

// Runtime length-scale setter. Clamped to [0.25, 4.0].
void kokoro_set_length_scale(struct kokoro_context* ctx, float scale);

// Load a voice-pack GGUF (arch="kokoro-voice"). Returns 0 on success.
int kokoro_load_voice_pack(struct kokoro_context* ctx, const char* path);

// Override the espeak-ng language (default "en-us"). Returns 0 on success.
int kokoro_set_language(struct kokoro_context* ctx, const char* espeak_lang);

// Drop cached (lang, text) → IPA entries.
void kokoro_phoneme_cache_clear(struct kokoro_context* ctx);

// Tokenise a phoneme string (already-IPA) into the model's vocab.
// Returns malloc'd int32_t[*out_n] — caller frees with free().
int32_t* kokoro_phonemes_to_ids(struct kokoro_context* ctx, const char* phonemes, int* out_n);

// Synthesise text → 24 kHz mono float32 PCM. Runs espeak-ng first.
// Returns malloc'd buffer; caller frees with kokoro_pcm_free.
float* kokoro_synthesize(struct kokoro_context* ctx, const char* text, int* out_n_samples);

// Same as kokoro_synthesize but the input is already IPA — skips espeak-ng.
float* kokoro_synthesize_phonemes(struct kokoro_context* ctx, const char* phonemes, int* out_n_samples);

// Return the sample rate declared by the loaded model.
int kokoro_sample_rate(const struct kokoro_context* ctx);

void kokoro_pcm_free(float* pcm);

#ifdef __cplusplus
}
#endif

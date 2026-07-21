// src/core/align.h — duration-based alignment helpers for TTS pipelines.
//
// Every TTS with explicit duration prediction (FastSpeech, VITS,
// StyleTTS, Kokoro, qwen3-tts talker) needs the same primitive: take a
// per-token feature matrix and a per-token integer duration vector,
// and return a per-frame feature matrix where each token's features
// are repeated `dur[i]` times. PyTorch reference:
//
//   indices = torch.repeat_interleave(torch.arange(L), durations)
//   en      = features[:, indices]   # or features.T @ one_hot(indices, L)
//
// On device graphs the gather form is preferred: upload compact (D, L)
// features plus an int32 index vector of length T_frames, then
// ggml_get_rows(features, indices) → (D, T_frames). build_indices()
// constructs that index vector on the host after duration rounding.
//
// repeat_interleave() keeps the older CPU memcpy expansion for tests
// and reference comparisons. Output buffers are malloc'd — caller
// frees with `free()`.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace core_align {

// Build the gather index vector for duration alignment:
//   indices = repeat_interleave(arange(L), durations)
// Returns a malloc'd I32 buffer of length T_frames = sum(durations);
// writes T_frames to *out_T_frames. Returns nullptr (and
// *out_T_frames=0) on empty input or allocation failure.
static inline int32_t* build_indices(int L, const int* durations, int* out_T_frames) {
    int T = 0;
    for (int i = 0; i < L; i++)
        T += durations[i];
    if (out_T_frames)
        *out_T_frames = T;
    if (T <= 0)
        return nullptr;
    int32_t* out = (int32_t*)std::malloc((size_t)T * sizeof(int32_t));
    if (!out) {
        if (out_T_frames)
            *out_T_frames = 0;
        return nullptr;
    }
    int j = 0;
    for (int i = 0; i < L; i++) {
        const int n = durations[i];
        for (int k = 0; k < n; k++)
            out[j++] = (int32_t)i;
    }
    return out;
}

// Repeat-interleave a (D, L) channel-major feature matrix according to
// per-token integer durations. Returns a malloc'd (D, T_frames) F32
// buffer with T_frames = sum(durations); writes T_frames to
// *out_T_frames. Returns nullptr (and *out_T_frames=0) on empty input
// or allocation failure.
//
// Layout: features[i] is the i-th token's column at offset i·D in the
// flat buffer, matching ggml's ne=(D, L) storage. The output preserves
// the same channel-major layout. Equivalent to
// ggml_get_rows(features, build_indices(...)) on contiguous F32 data.
static inline float* repeat_interleave(const float* features, int D, int L, const int* durations, int* out_T_frames) {
    int T = 0;
    for (int i = 0; i < L; i++)
        T += durations[i];
    if (out_T_frames)
        *out_T_frames = T;
    if (T <= 0)
        return nullptr;
    float* out = (float*)std::malloc((size_t)D * T * sizeof(float));
    if (!out)
        return nullptr;
    int j = 0;
    for (int i = 0; i < L; i++) {
        const int n = durations[i];
        const float* col = features + (size_t)i * D;
        for (int k = 0; k < n; k++) {
            std::memcpy(out + (size_t)j * D, col, (size_t)D * sizeof(float));
            j++;
        }
    }
    return out;
}

} // namespace core_align

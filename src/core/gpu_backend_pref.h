// gpu_backend_pref.h — CPU-only stub for kokoro.cpp standalone build.
// CrispASR uses this to select Metal/CUDA/Vulkan; we always return nullptr.
#pragma once
#include "ggml-backend.h"

static inline ggml_backend_t crispasr_init_gpu_backend() {
    return nullptr;
}

// core/gpu_backend_pref.h — CPU-only stub for kokoro.cpp standalone build.
// Always returns nullptr; GPU backends are not supported in this project.
#pragma once
#include "ggml-backend.h"

static inline ggml_backend_t kokoro_init_gpu_backend() {
    return nullptr;
}

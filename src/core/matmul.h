#pragma once

#include "ggml.h"

namespace core_ggml {

// Kokoro's recurrent and phase-producing stages amplify low-precision
// accumulation error. Request F32 accumulation consistently on GPU backends.
static inline ggml_tensor* mul_mat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    ggml_tensor* out = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return out;
}

} // namespace core_ggml

// src/core/m_source_host.h — host-side m_source linear + tanh.
//
// The generator harmonic path applies a tiny immutable Linear(9→1)+tanh to the
// sine stack. Those weights are cached as host floats at context init so
// harmonic generation never touches ggml backends concurrently with scheduler
// compute. Helpers here are model-free and side-effect free.

#pragma once

#include <cmath>
namespace core_m_source {

constexpr int kDim = 9;

// y = tanh(bias + dot(x, w)) for one time step.
inline float apply(const float* x, const float* w, float bias) {
    float s = bias;
    for (int k = 0; k < kDim; k++)
        s += x[k] * w[k];
    return std::tanh(s);
}

} // namespace core_m_source

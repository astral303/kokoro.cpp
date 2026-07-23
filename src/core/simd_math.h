#pragma once

#include <cstddef>
#include <cstdint>

namespace core_simd {

void sin_scaled_inplace(float * values, std::size_t count, float scale);

void fill_standard_normal(
    float * values, std::size_t count, std::uint64_t seed);

void stft20(
    const float * padded,
    std::size_t frame_count,
    std::size_t hop,
    const float * hann,
    const float * twiddle_cos_transposed,
    const float * twiddle_sin_transposed,
    float * output);

} // namespace core_simd

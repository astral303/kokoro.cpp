#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "core/simd_math.cpp"
#include "hwy/foreach_target.h"

#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"

#include "core/simd_math.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

HWY_BEFORE_NAMESPACE();
namespace core_simd {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void SinScaledInplace(float * values, std::size_t count, float scale) {
    const hn::ScalableTag<float> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto scale_vec = hn::Set(d, scale);
    std::size_t i = 0;
    for (; i + lanes <= count; i += lanes) {
        const auto value = hn::LoadU(d, values + i);
        hn::StoreU(hn::Mul(hn::Sin(d, value), scale_vec), d, values + i);
    }
    for (; i < count; ++i) {
        values[i] = std::sin(values[i]) * scale;
    }
}

void FillStandardNormal(
    float * values, std::size_t count, std::uint64_t seed) {
#if HWY_HAVE_FLOAT64
    hn::VectorXoshiro rng(seed);
    const hn::ScalableTag<double> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto minus_two = hn::Set(d, -2.0);
    const auto two_pi = hn::Set(d, 6.28318530717958647692);
    const auto minimum = hn::Set(d, 0x1.0p-53);
    std::vector<double> first(lanes);
    std::vector<double> second(lanes);
    std::size_t output = 0;
    while (output < count) {
        const auto uniform_radius = hn::Max(rng.Uniform(), minimum);
        const auto uniform_angle = rng.Uniform();
        const auto radius =
            hn::Sqrt(hn::Mul(minus_two, hn::Log(d, uniform_radius)));
        const auto angle = hn::Mul(two_pi, uniform_angle);
        auto sine = hn::Zero(d);
        auto cosine = hn::Zero(d);
        hn::SinCos(d, angle, sine, cosine);
        hn::StoreU(hn::Mul(radius, cosine), d, first.data());
        hn::StoreU(hn::Mul(radius, sine), d, second.data());
        for (std::size_t lane = 0; lane < lanes && output < count; ++lane) {
            values[output++] = static_cast<float>(first[lane]);
            if (output < count) {
                values[output++] = static_cast<float>(second[lane]);
            }
        }
    }
#else
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = normal(rng);
    }
#endif
}

void Stft20(
    const float * padded,
    std::size_t frame_count,
    std::size_t hop,
    const float * hann,
    const float * twiddle_cos_transposed,
    const float * twiddle_sin_transposed,
    float * output) {
    constexpr std::size_t n_fft = 20;
    constexpr std::size_t n_bins = 11;
    constexpr std::size_t output_channels = 22;
    const hn::ScalableTag<float> d;
    const std::size_t lanes = hn::Lanes(d);
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        float windowed[n_fft];
        const float * input = padded + frame * hop;
        for (std::size_t n = 0; n < n_fft; ++n) {
            windowed[n] = input[n] * hann[n];
        }
        for (std::size_t bin = 0; bin < n_bins; bin += lanes) {
            const std::size_t active = std::min(lanes, n_bins - bin);
            auto real = hn::Zero(d);
            auto imag = hn::Zero(d);
            for (std::size_t n = 0; n < n_fft; ++n) {
                const auto value = hn::Set(d, windowed[n]);
                const auto cosine = hn::LoadN(
                    d, twiddle_cos_transposed + n * n_bins + bin, active);
                const auto sine = hn::LoadN(
                    d, twiddle_sin_transposed + n * n_bins + bin, active);
                real = hn::Add(real, hn::Mul(value, cosine));
                imag = hn::Add(imag, hn::Mul(value, sine));
            }
            const auto magnitude = hn::Sqrt(
                hn::Add(hn::Mul(real, real), hn::Mul(imag, imag)));
            const auto phase = hn::Atan2(d, imag, real);
            hn::StoreN(
                magnitude, d, output + frame * output_channels + bin, active);
            hn::StoreN(
                phase, d,
                output + frame * output_channels + n_bins + bin, active);
        }
    }
}

} // namespace HWY_NAMESPACE
} // namespace core_simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace core_simd {

HWY_EXPORT(SinScaledInplace);
HWY_EXPORT(FillStandardNormal);
HWY_EXPORT(Stft20);

void sin_scaled_inplace(float * values, std::size_t count, float scale) {
    HWY_DYNAMIC_DISPATCH(SinScaledInplace)(values, count, scale);
}

void fill_standard_normal(
    float * values, std::size_t count, std::uint64_t seed) {
    HWY_DYNAMIC_DISPATCH(FillStandardNormal)(values, count, seed);
}

void stft20(
    const float * padded,
    std::size_t frame_count,
    std::size_t hop,
    const float * hann,
    const float * twiddle_cos_transposed,
    const float * twiddle_sin_transposed,
    float * output) {
    HWY_DYNAMIC_DISPATCH(Stft20)(
        padded, frame_count, hop, hann,
        twiddle_cos_transposed, twiddle_sin_transposed, output);
}

} // namespace core_simd
#endif

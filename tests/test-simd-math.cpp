#include "core/simd_math.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr float kTwoPi = 6.283185307179586f;

bool test_sine() {
    std::vector<float> values(10003);
    std::vector<float> expected(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = -1000000.0f + static_cast<float>(i) * 199.98001f;
        expected[i] = std::sin(values[i]) * 0.1f;
    }
    core_simd::sin_scaled_inplace(values.data(), values.size(), 0.1f);
    float max_error = 0.0f;
    for (size_t i = 0; i < values.size(); ++i) {
        max_error = std::max(max_error, std::fabs(values[i] - expected[i]));
    }
    if (max_error > 0.005f) {
        std::fprintf(stderr, "SIMD sine max error is %g\n", max_error);
        return false;
    }
    return true;
}

bool test_stft() {
    constexpr size_t n_fft = 20;
    constexpr size_t n_bins = 11;
    constexpr size_t frames = 17;
    constexpr size_t hop = 5;
    std::vector<float> input((frames - 1) * hop + n_fft);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = 0.2f * std::sin(0.17f * static_cast<float>(i)) +
            0.1f * std::cos(0.071f * static_cast<float>(i));
    }
    float hann[n_fft];
    float twiddle_cos[n_fft * n_bins];
    float twiddle_sin[n_fft * n_bins];
    for (size_t n = 0; n < n_fft; ++n) {
        hann[n] = 0.5f -
            0.5f * std::cos(kTwoPi * static_cast<float>(n) / n_fft);
        for (size_t bin = 0; bin < n_bins; ++bin) {
            const float angle = -kTwoPi * static_cast<float>(bin * n) / n_fft;
            twiddle_cos[n * n_bins + bin] = std::cos(angle);
            twiddle_sin[n * n_bins + bin] = std::sin(angle);
        }
    }
    std::vector<float> output(frames * 2 * n_bins);
    core_simd::stft20(
        input.data(), frames, hop, hann,
        twiddle_cos, twiddle_sin, output.data());
    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t bin = 0; bin < n_bins; ++bin) {
            float real = 0.0f;
            float imag = 0.0f;
            for (size_t n = 0; n < n_fft; ++n) {
                const float value = input[frame * hop + n] * hann[n];
                real += value * twiddle_cos[n * n_bins + bin];
                imag += value * twiddle_sin[n * n_bins + bin];
            }
            const float expected_magnitude = std::sqrt(real * real + imag * imag);
            const float expected_phase = std::atan2(imag, real);
            const float magnitude = output[frame * 2 * n_bins + bin];
            const float phase = output[frame * 2 * n_bins + n_bins + bin];
            if (std::fabs(magnitude - expected_magnitude) > 1.0e-5f ||
                (expected_magnitude > 1.0e-5f &&
                 std::fabs(phase - expected_phase) > 1.0e-4f)) {
                std::fprintf(
                    stderr, "SIMD STFT differs at frame=%zu bin=%zu\n",
                    frame, bin);
                return false;
            }
        }
    }
    return true;
}

bool test_normal() {
    std::vector<float> first(100000);
    std::vector<float> second(first.size());
    core_simd::fill_standard_normal(first.data(), first.size(), 1234);
    core_simd::fill_standard_normal(second.data(), second.size(), 1234);
    if (first != second) {
        std::fprintf(stderr, "SIMD normal generation is not repeatable\n");
        return false;
    }
    double sum = 0.0;
    double sum_squares = 0.0;
    for (float value : first) {
        sum += value;
        sum_squares += static_cast<double>(value) * value;
    }
    const double mean = sum / first.size();
    const double variance = sum_squares / first.size() - mean * mean;
    if (std::fabs(mean) > 0.02 || std::fabs(variance - 1.0) > 0.03) {
        std::fprintf(
            stderr, "SIMD normal moments differ: mean=%g variance=%g\n",
            mean, variance);
        return false;
    }
    return true;
}

} // namespace

int main() {
    return test_sine() && test_stft() && test_normal() ? 0 : 1;
}

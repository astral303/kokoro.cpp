// test-align.cpp — model-free duration-alignment helper tests.

#include "core/align.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(condition) do {                                                   \
    ++g_checks;                                                                  \
    if (!(condition)) {                                                         \
        ++g_failures;                                                            \
        std::fprintf(stderr, "FAILED: %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    }                                                                            \
} while (0)

static void test_indices_match_repeat_interleave() {
    constexpr int D = 2;
    constexpr int L = 3;
    const int durations[L] = {2, 1, 3};
    const float features[D * L] = {
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
    };
    const int32_t expected_indices[6] = {0, 0, 1, 2, 2, 2};

    int index_frames = 0;
    int32_t* indices = core_align::build_indices(L, durations, &index_frames);
    CHECK(indices != nullptr);
    CHECK(index_frames == 6);
    for (int i = 0; indices && i < index_frames; ++i) {
        CHECK(indices[i] == expected_indices[i]);
    }

    int expanded_frames = 0;
    float* expanded = core_align::repeat_interleave(
        features, D, L, durations, &expanded_frames);
    CHECK(expanded != nullptr);
    CHECK(expanded_frames == index_frames);
    for (int frame = 0; indices && expanded && frame < index_frames; ++frame) {
        for (int channel = 0; channel < D; ++channel) {
            const float gathered = features[(size_t)indices[frame] * D + channel];
            CHECK(expanded[(size_t)frame * D + channel] == gathered);
        }
    }

    std::free(indices);
    std::free(expanded);
}

static void test_empty_alignment() {
    const int durations[2] = {0, 0};
    int frames = 99;
    int32_t* indices = core_align::build_indices(2, durations, &frames);
    CHECK(indices == nullptr);
    CHECK(frames == 0);
}

int main() {
    test_indices_match_repeat_interleave();
    test_empty_alignment();
    std::fprintf(stderr, "%d alignment checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

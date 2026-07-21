// test-smoke.cpp — model-free C API smoke tests for kokoro.cpp.

#include "kokoro.h"
#include <cstdio>
#include <cstring>

static int g_tests = 0;
static int g_failed = 0;

#define CHECK(condition) do {                                                   \
    ++g_tests;                                                                  \
    if (!(condition)) {                                                         \
        ++g_failed;                                                             \
        std::fprintf(stderr, "FAILED: %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    }                                                                           \
} while (0)

static void test_default_params() {
    struct kokoro_context_params p = kokoro_context_default_params();
    CHECK(p.n_threads > 0);
    CHECK(p.length_scale == 1.0f);
    CHECK(p.verbosity == 1);
    CHECK(std::strcmp(p.espeak_lang, "en-us") == 0);
}

static void test_null_model() {
    struct kokoro_context *ctx = kokoro_init_from_file(NULL, kokoro_context_default_params());
    CHECK(ctx == NULL);
}

static void test_missing_model() {
    struct kokoro_context_params p = kokoro_context_default_params();
    p.verbosity = 0;
    struct kokoro_context *ctx = kokoro_init_from_file("/nonexistent/model.gguf", p);
    CHECK(ctx == NULL);
}

static void test_null_guards() {
    kokoro_free(NULL);
    kokoro_pcm_free(NULL);
    int n = 0;
    int32_t *ids = kokoro_phonemes_to_ids(NULL, "hello", &n);
    CHECK(ids == NULL);
    CHECK(n == 0);
    CHECK(kokoro_sample_rate(NULL) == 0);
}

int main() {
    fprintf(stderr, "kokoro.cpp smoke tests (model-free)\n");

    test_default_params();
    test_null_model();
    test_missing_model();
    test_null_guards();

    fprintf(stderr, "%d checks, %d failures\n", g_tests, g_failed);
    return g_failed == 0 ? 0 : 1;
}

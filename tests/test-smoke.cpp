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

static void test_version() {
    const char* v = kokoro_version();
    CHECK(v != NULL);
    CHECK(std::strstr(v, "kokoro.cpp") != NULL);
}

static void test_default_params() {
    struct kokoro_context_params p = kokoro_context_default_params();
    CHECK(p.abi_version == KOKORO_ABI_VERSION);
    CHECK(p.n_threads > 0);
    CHECK(p.length_scale == 1.0f);
    CHECK(p.verbosity == 1);
    CHECK(std::strcmp(p.espeak_lang, "en-us") == 0);
}

static void test_abi_version_check() {
    struct kokoro_context_params p = kokoro_context_default_params();
    p.abi_version = KOKORO_ABI_VERSION + 99; // future version
    struct kokoro_context* ctx = kokoro_init_from_file("dummy.gguf", p);
    CHECK(ctx == NULL);
    const char* err = kokoro_last_error();
    CHECK(err != NULL);
    CHECK(std::strstr(err, "abi_version") != NULL);

    p.abi_version = 0;
    ctx = kokoro_init_from_file("dummy.gguf", p);
    CHECK(ctx == NULL);
    CHECK(std::strstr(kokoro_last_error(), "abi_version") != NULL);
}

static void test_null_model() {
    struct kokoro_context* ctx = kokoro_init_from_file(NULL, kokoro_context_default_params());
    CHECK(ctx == NULL);
    CHECK(std::strlen(kokoro_last_error()) > 0);
}

static void test_missing_model() {
    struct kokoro_context_params p = kokoro_context_default_params();
    p.verbosity = 0;
    struct kokoro_context* ctx = kokoro_init_from_file("/nonexistent/model.gguf", p);
    CHECK(ctx == NULL);
}

static void test_null_guards() {
    kokoro_free(NULL);
    kokoro_pcm_free(NULL);
    int n = 0;
    int32_t* ids = kokoro_phonemes_to_ids(NULL, "hello", &n);
    CHECK(ids == NULL);
    CHECK(n == 0);
    CHECK(kokoro_sample_rate(NULL) == 0);
    CHECK(kokoro_voice_name(NULL) == NULL);
    CHECK(kokoro_voice_max_phonemes(NULL) == 0);
}

static void test_audio_struct() {
    struct kokoro_audio a = {};
    CHECK(a.samples == NULL);
    CHECK(a.n_samples == 0);
    kokoro_audio_free(&a); // safe on zero-init
    kokoro_audio_free(NULL); // safe on NULL
}

static void test_structured_synth_null() {
    struct kokoro_audio a = {};
    enum kokoro_status s = kokoro_synthesize_audio(NULL, "hello", &a);
    CHECK(s == KOKORO_STATUS_INVALID_PARAMS);
    s = kokoro_synthesize_phonemes_audio(NULL, "hello", &a);
    CHECK(s == KOKORO_STATUS_INVALID_PARAMS);
}

static int g_log_count = 0;
static void test_log_cb(enum kokoro_log_level, const char*, void* ud) {
    (void)ud;
    g_log_count++;
}

static void test_log_callback() {
    kokoro_log_set(test_log_cb, NULL);
    // Trigger a log by trying to init with bad path
    struct kokoro_context_params p = kokoro_context_default_params();
    p.verbosity = 0;
    struct kokoro_context* ctx = kokoro_init_from_file("/x.gguf", p);
    (void)ctx;
    CHECK(g_log_count > 0);
    kokoro_log_set(NULL, NULL); // restore default
}

int main() {
    std::fprintf(stderr, "kokoro.cpp smoke tests (model-free)\n");

    test_version();
    test_default_params();
    test_abi_version_check();
    test_null_model();
    test_missing_model();
    test_null_guards();
    test_audio_struct();
    test_structured_synth_null();
    test_log_callback();

    std::fprintf(stderr, "%d checks, %d failures\n", g_tests, g_failed);
    return g_failed == 0 ? 0 : 1;
}

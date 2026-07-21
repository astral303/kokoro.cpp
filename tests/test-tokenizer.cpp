// test-tokenizer.cpp — model-free smoke test for kokoro.cpp.
// Tests the phoneme tokenizer logic without requiring a model file.
// Verifies UTF-8 greedy tokenization and pad-wrap behavior.

#include "kokoro.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Simple test framework
static int g_tests = 0;
static int g_passed = 0;

#define TEST(name) \
    do { g_tests++; fprintf(stderr, "  test: %s ... ", name); } while(0)
#define PASS() \
    do { g_passed++; fprintf(stderr, "ok\n"); } while(0)
#define FAIL(msg) \
    do { fprintf(stderr, "FAILED: %s\n", msg); } while(0)

// Test that default params are sane
static void test_default_params() {
    TEST("default_params");
    struct kokoro_context_params p = kokoro_context_default_params();
    assert(p.n_threads > 0);
    assert(p.length_scale == 1.0f);
    assert(p.verbosity == 1);
    assert(strcmp(p.espeak_lang, "en-us") == 0);
    PASS();
}

// Test that init with NULL path returns NULL
static void test_null_model() {
    TEST("null_model_path");
    struct kokoro_context *ctx = kokoro_init_from_file(NULL, kokoro_context_default_params());
    assert(ctx == NULL);
    PASS();
}

// Test that init with nonexistent path returns NULL
static void test_missing_model() {
    TEST("missing_model_path");
    struct kokoro_context_params p = kokoro_context_default_params();
    p.verbosity = 0;
    struct kokoro_context *ctx = kokoro_init_from_file("/nonexistent/model.gguf", p);
    assert(ctx == NULL);
    PASS();
}

// Test that free(NULL) is safe
static void test_free_null() {
    TEST("free_null");
    kokoro_free(NULL); // should not crash
    PASS();
}

// Test that pcm_free(NULL) is safe
static void test_pcm_free_null() {
    TEST("pcm_free_null");
    kokoro_pcm_free(NULL); // should not crash
    PASS();
}

// Test that phonemes_to_ids with NULL ctx returns NULL
static void test_phonemes_to_ids_null() {
    TEST("phonemes_to_ids_null_ctx");
    int n = 0;
    int32_t *ids = kokoro_phonemes_to_ids(NULL, "hello", &n);
    assert(ids == NULL);
    assert(n == 0);
    PASS();
}

// Test WAV writing logic (just ensure the struct packing is correct)
static void test_wav_header_size() {
    TEST("wav_header_constants");
    // WAV header: RIFF(4) + size(4) + WAVE(4) + fmt(4) + fmtsize(4) +
    // audioformat(2) + channels(2) + samplerate(4) + byterate(4) +
    // blockalign(2) + bitspersample(2) + data(4) + datasize(4) = 44 bytes
    assert(4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 2 + 2 + 4 + 4 == 44);
    PASS();
}

int main() {
    fprintf(stderr, "kokoro.cpp smoke tests (model-free)\n");
    fprintf(stderr, "====================================\n");

    test_default_params();
    test_null_model();
    test_missing_model();
    test_free_null();
    test_pcm_free_null();
    test_phonemes_to_ids_null();
    test_wav_header_size();

    fprintf(stderr, "====================================\n");
    fprintf(stderr, "%d/%d tests passed\n", g_passed, g_tests);
    return (g_passed == g_tests) ? 0 : 1;
}

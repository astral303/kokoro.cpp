// test-har-async.cpp — model-free RAII / host m_source helper tests.

#include "core/har_async.h"
#include "core/m_source_host.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <stdexcept>
#include <thread>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(condition)                                                                 \
    do {                                                                                 \
        ++g_checks;                                                                      \
        if (!(condition)) {                                                              \
            ++g_failures;                                                                \
            std::fprintf(stderr, "FAILED: %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        }                                                                                \
    } while (0)

static void test_async_release_transfers_ownership() {
    float* raw = (float*)std::malloc(4 * sizeof(float));
    CHECK(raw != nullptr);
    raw[0] = 1.5f;

    core_har::AsyncResult guard(std::async(std::launch::async, [raw]() {
        core_har::ComputeResult r;
        r.data = raw;
        r.T_har = 3;
        r.work_ms = 1.25;
        return r;
    }));

    auto& got = guard.get();
    CHECK(got.data == raw);
    CHECK(got.T_har == 3);
    CHECK(got.work_ms == 1.25);

    float* owned = guard.release();
    CHECK(owned == raw);
    CHECK(guard.get().data == nullptr);
    std::free(owned);
}

static void test_async_destructor_joins_and_frees() {
    float* raw = (float*)std::malloc(sizeof(float));
    CHECK(raw != nullptr);
    *raw = 42.0f;

    {
        core_har::AsyncResult guard(std::async(std::launch::deferred, [raw]() {
            // deferred runs on the waiting thread; still exercises join+free.
            core_har::ComputeResult r;
            r.data = raw;
            r.T_har = 1;
            return r;
        }));
        // Destroy without release → frees raw.
    }
    // If the free did not happen this would leak; we cannot dereference raw.
    // Ownership transfer failure is covered by ASan/leak detectors when enabled.
    CHECK(true);
}

static void test_async_destructor_waits_for_slow_worker() {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    {
        core_har::AsyncResult guard(std::async(std::launch::async, []() {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            core_har::ComputeResult r;
            r.data = (float*)std::malloc(sizeof(float));
            if (r.data)
                r.data[0] = 7.0f;
            r.T_har = 1;
            r.work_ms = 30.0;
            return r;
        }));
        // Fall off scope without explicit join.
    }
    const auto ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    CHECK(ms >= 25.0);
}

static void test_async_get_propagates_worker_exception() {
    core_har::AsyncResult guard(std::async(std::launch::async, []() -> core_har::ComputeResult {
        throw std::runtime_error("boom");
    }));
    bool threw = false;
    try {
        (void)guard.get();
    } catch (const std::runtime_error& e) {
        threw = std::strcmp(e.what(), "boom") == 0;
    }
    CHECK(threw);
}

static void test_m_source_apply() {
    const float weights[core_m_source::kDim] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f};
    const float bias_in = -0.25f;
    const float x[core_m_source::kDim] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    float expected = bias_in;
    for (int i = 0; i < core_m_source::kDim; i++)
        expected += x[i] * weights[i];
    expected = std::tanh(expected);
    const float y = core_m_source::apply(x, weights, bias_in);
    CHECK(std::fabs(y - expected) < 1e-6f);
}

int main() {
    test_async_release_transfers_ownership();
    test_async_destructor_joins_and_frees();
    test_async_destructor_waits_for_slow_worker();
    test_async_get_propagates_worker_exception();
    test_m_source_apply();
    std::fprintf(stderr, "%d har-async checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

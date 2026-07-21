// src/core/har_async.h — RAII owner for async harmonic-source results.
//
// Harmonic generation can run on a worker while Vulkan executes the text
// encoder and decoder. The worker returns a malloc'd buffer plus timing
// metadata; this guard always joins and frees on destruction so every early
// return path stays leak- and race-free. release() transfers buffer ownership
// to the caller (typically the generator upload).

#pragma once

#include <cstdlib>
#include <future>
#include <utility>

namespace core_har {

struct ComputeResult {
    float* data = nullptr;
    int T_har = 0;
    double work_ms = 0.0;
};

// Joins a future<ComputeResult> and frees the malloc'd payload unless released.
// Destructor is noexcept: worker exceptions are swallowed there so stack
// unwinding through the C ABI cannot escape. Callers that need failure
// signalling must join()/get() explicitly on the success path.
struct AsyncResult {
    std::future<ComputeResult> fut;
    ComputeResult local{};
    bool joined = false;

    AsyncResult() = default;
    explicit AsyncResult(std::future<ComputeResult>&& f) : fut(std::move(f)) {}

    AsyncResult(const AsyncResult&) = delete;
    AsyncResult& operator=(const AsyncResult&) = delete;

    AsyncResult(AsyncResult&& o) noexcept
        : fut(std::move(o.fut)), local(o.local), joined(o.joined) {
        o.local = {};
        o.joined = true;
    }

    AsyncResult& operator=(AsyncResult&& o) noexcept {
        if (this == &o)
            return *this;
        reset();
        fut = std::move(o.fut);
        local = o.local;
        joined = o.joined;
        o.local = {};
        o.joined = true;
        return *this;
    }

    ~AsyncResult() { reset(); }

    bool valid() const { return fut.valid() || joined; }

    void join() {
        if (joined)
            return;
        if (fut.valid()) {
            try {
                local = fut.get();
            } catch (...) {
                joined = true;
                throw;
            }
        }
        joined = true;
    }

    ComputeResult& get() {
        join();
        return local;
    }

    // Join and transfer ownership of local.data to the caller. Subsequent
    // destruction will not free the buffer.
    float* release() {
        join();
        float* p = local.data;
        local.data = nullptr;
        return p;
    }

    void reset() noexcept {
        if (!joined && fut.valid()) {
            try {
                local = fut.get();
            } catch (...) {
                local = {};
            }
            joined = true;
        }
        std::free(local.data);
        local.data = nullptr;
        local.T_har = 0;
        local.work_ms = 0.0;
    }
};

} // namespace core_har

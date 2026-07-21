// core/kokoro_env.h — simple environment variable access for kokoro.cpp.
// All user-facing env vars use the KOKORO_ prefix.
#pragma once

#include <cstdlib>
#include <cstring>

namespace kokoro_env {

// Read env var. Returns nullptr if unset.
inline const char* get(const char* name) {
    return std::getenv(name);
}

// True if set, non-empty, and not "0".
inline bool truthy(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

} // namespace kokoro_env

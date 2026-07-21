#pragma once

#include "kokoro.h"

// Internal diagnostics use the public callback and are silent when no
// callback is installed. Library code must never write directly to stderr.
void kokoro_log_internal(enum kokoro_log_level level, const char* fmt, ...);

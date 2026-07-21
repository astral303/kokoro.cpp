cmake_minimum_required(VERSION 3.14)
project(kokoro.cpp VERSION 0.1.0 LANGUAGES C CXX)

# Generate version info
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE KOKORO_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT KOKORO_GIT_HASH)
    set(KOKORO_GIT_HASH "unknown")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ===========================================================================
# Options
# ===========================================================================

option(KOKORO_BUILD_CLI    "Build the kokoro-cli tool"    ON)
option(KOKORO_BUILD_TESTS  "Build tests"                  ON)
option(KOKORO_STATIC       "Build kokoro as a static lib" ON)

# ===========================================================================
# ggml (CPU-only)
# ===========================================================================

set(GGML_METAL  OFF CACHE BOOL "" FORCE)
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_SYCL   OFF CACHE BOOL "" FORCE)
set(GGML_RPC    OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(ggml)

# ===========================================================================
# espeak-ng (required, directly linked)
# ===========================================================================

# Try pkg-config first (works with Nix, system packages)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ESPEAK_NG IMPORTED_TARGET espeak-ng)
endif()

if(NOT ESPEAK_NG_FOUND)
    # Fallback: find the library and header manually
    find_path(ESPEAK_NG_INCLUDEDIR espeak-ng/speak_lib.h
        PATHS /usr/include /usr/local/include /opt/homebrew/include
        PATH_SUFFIXES include)
    find_library(ESPEAK_NG_LIB NAMES espeak-ng espeak
        PATHS /usr/lib /usr/local/lib /opt/homebrew/lib
              /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
        PATH_SUFFIXES lib)

    if(ESPEAK_NG_INCLUDEDIR AND ESPEAK_NG_LIB)
        # Create an imported target so downstream usage is uniform
        add_library(PkgConfig::ESPEAK_NG UNKNOWN IMPORTED)
        set_target_properties(PkgConfig::ESPEAK_NG PROPERTIES
            IMPORTED_LOCATION "${ESPEAK_NG_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ESPEAK_NG_INCLUDEDIR}")
        set(ESPEAK_NG_FOUND TRUE)
    endif()
endif()

if(NOT ESPEAK_NG_FOUND)
    message(FATAL_ERROR
        "espeak-ng is REQUIRED but was not found.\n"
        "Install it via your package manager:\n"
        "  Debian/Ubuntu: sudo apt install libespeak-ng-dev\n"
        "  Fedora:        sudo dnf install espeak-ng-devel\n"
        "  macOS:         brew install espeak-ng\n"
        "  Nix:           available via parent flake\n"
        "Or set ESPEAK_NG_INCLUDEDIR and ESPEAK_NG_LIB manually.")
endif()

message(STATUS "espeak-ng found: ${ESPEAK_NG_FOUND}")

# ===========================================================================
# kokoro library
# ===========================================================================

set(KOKORO_SOURCES
    src/kokoro.cpp
    src/core/gguf_loader.cpp
)

if(KOKORO_STATIC)
    add_library(kokoro STATIC ${KOKORO_SOURCES})
else()
    add_library(kokoro SHARED ${KOKORO_SOURCES})
endif()

target_include_directories(kokoro PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(kokoro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src
)

target_link_libraries(kokoro PUBLIC
    ggml
    PkgConfig::ESPEAK_NG
)

# Strict warnings for project-owned code (not propagated to ggml)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kokoro PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

target_compile_definitions(kokoro PRIVATE
    KOKORO_BUILD
    KOKORO_VERSION="${PROJECT_VERSION}"
    KOKORO_GIT_HASH="${KOKORO_GIT_HASH}"
)

# Propagate KOKORO_STATIC to consumers of the static lib
if(KOKORO_STATIC)
    target_compile_definitions(kokoro PUBLIC KOKORO_STATIC)
endif()

# ===========================================================================
# CLI tool
# ===========================================================================

if(KOKORO_BUILD_CLI)
    add_executable(kokoro-cli tools/kokoro-cli.cpp)
    target_link_libraries(kokoro-cli PRIVATE kokoro)
    target_include_directories(kokoro-cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kokoro-cli PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

# ===========================================================================
# Tests
# ===========================================================================

if(KOKORO_BUILD_TESTS)
    enable_testing()

    add_executable(test-smoke tests/test-smoke.cpp)
    target_link_libraries(test-smoke PRIVATE kokoro)
    target_include_directories(test-smoke PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test-smoke PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME smoke COMMAND test-smoke)

    add_executable(test-c-api tests/test-c-api.c)
    target_link_libraries(test-c-api PRIVATE kokoro)
    set_target_properties(test-c-api PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME c-api COMMAND test-c-api)
endif()

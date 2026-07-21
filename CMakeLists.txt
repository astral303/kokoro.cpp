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

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(KOKORO_STANDALONE ON)
else()
    set(KOKORO_STANDALONE OFF)
endif()

if(APPLE)
    set(_KOKORO_DEFAULT_VULKAN OFF)
    set(_KOKORO_DEFAULT_METAL ON)
else()
    set(_KOKORO_DEFAULT_VULKAN ON)
    set(_KOKORO_DEFAULT_METAL OFF)
endif()

if(KOKORO_STANDALONE)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(KOKORO_STANDALONE AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ===========================================================================
# Options
# ===========================================================================

option(KOKORO_BUILD_CLI    "Build the kokoro-cli tool"    ${KOKORO_STANDALONE})
option(KOKORO_BUILD_TESTS  "Build tests"                  ${KOKORO_STANDALONE})
option(KOKORO_STATIC       "Build kokoro as a static lib" ON)
option(KOKORO_WERROR       "Treat warnings in kokoro.cpp sources as errors" OFF)
option(KOKORO_VULKAN       "Enable Vulkan GPU backend" ${_KOKORO_DEFAULT_VULKAN})
option(KOKORO_METAL        "Enable Metal GPU backend" ${_KOKORO_DEFAULT_METAL})
option(KOKORO_CUDA         "Enable CUDA GPU backend" OFF)
set(KOKORO_GGML_DIR "" CACHE PATH
    "External ggml source checkout (default: bundled submodule)")

# ===========================================================================
# ggml (CPU-only)
# ===========================================================================

function(_kokoro_add_ggml source_dir)
    # Keep dependency settings local to this function so embedding kokoro.cpp
    # does not overwrite a parent project's GGML_* or BUILD_SHARED_LIBS values.
    set(GGML_BUILD_TESTS OFF)
    set(GGML_BUILD_EXAMPLES OFF)
    set(GGML_CPU ON)
    set(GGML_METAL ${KOKORO_METAL})
    set(GGML_METAL_EMBED_LIBRARY ${KOKORO_METAL})
    set(GGML_CUDA ${KOKORO_CUDA})
    set(GGML_VULKAN ${KOKORO_VULKAN})
    set(GGML_SYCL OFF)
    set(GGML_RPC OFF)
    set(GGML_OPENCL OFF)
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_subdirectory("${source_dir}" "${CMAKE_CURRENT_BINARY_DIR}/ggml"
        EXCLUDE_FROM_ALL)
endfunction()

if(TARGET ggml)
    message(STATUS "kokoro.cpp: using parent-provided ggml target")
    if(NOT KOKORO_GGML_DIR)
        get_target_property(_KOKORO_GGML_TARGET_DIR ggml SOURCE_DIR)
        if(EXISTS "${_KOKORO_GGML_TARGET_DIR}/src/ggml-backend-impl.h")
            set(KOKORO_GGML_DIR "${_KOKORO_GGML_TARGET_DIR}")
        elseif(EXISTS "${_KOKORO_GGML_TARGET_DIR}/ggml-backend-impl.h")
            get_filename_component(KOKORO_GGML_DIR
                "${_KOKORO_GGML_TARGET_DIR}" DIRECTORY)
        endif()
    endif()
else()
    if(NOT KOKORO_GGML_DIR)
        set(KOKORO_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
        message(STATUS "kokoro.cpp: using bundled ggml submodule")
    else()
        get_filename_component(KOKORO_GGML_DIR "${KOKORO_GGML_DIR}"
            ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        message(STATUS "kokoro.cpp: using external ggml checkout: ${KOKORO_GGML_DIR}")
    endif()

    if(NOT EXISTS "${KOKORO_GGML_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "KOKORO_GGML_DIR is not a ggml source checkout: ${KOKORO_GGML_DIR}")
    endif()
    _kokoro_add_ggml("${KOKORO_GGML_DIR}")
endif()

# Backends may have been created by kokoro.cpp or supplied by the parent.
set(KOKORO_HAS_VULKAN OFF)
set(KOKORO_HAS_METAL OFF)
set(KOKORO_HAS_CUDA OFF)
if(KOKORO_VULKAN AND TARGET ggml-vulkan)
    set(KOKORO_HAS_VULKAN ON)
    message(STATUS "kokoro.cpp Vulkan backend: enabled")
endif()
if(KOKORO_METAL AND TARGET ggml-metal)
    set(KOKORO_HAS_METAL ON)
    message(STATUS "kokoro.cpp Metal backend: enabled")
endif()
if(KOKORO_CUDA AND TARGET ggml-cuda)
    set(KOKORO_HAS_CUDA ON)
    message(STATUS "kokoro.cpp CUDA backend: enabled")
endif()

# core/gguf_loader.cpp uses one private ggml backend header in addition to
# ggml's public target includes. A source checkout is therefore required even
# when the parent already provides the target.
if(NOT KOKORO_GGML_DIR OR
   NOT EXISTS "${KOKORO_GGML_DIR}/src/ggml-backend-impl.h")
    message(FATAL_ERROR
        "Cannot locate ggml private headers. Set KOKORO_GGML_DIR to the source "
        "checkout used by the parent-provided ggml target.")
endif()

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
add_library(kokoro::kokoro ALIAS kokoro)
set_target_properties(kokoro PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
if(NOT KOKORO_STATIC AND UNIX AND NOT APPLE)
    target_link_options(kokoro PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/kokoro.map")
endif()

target_include_directories(kokoro PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(kokoro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${KOKORO_GGML_DIR}/src
)

target_link_libraries(kokoro PUBLIC
    ggml
    PkgConfig::ESPEAK_NG
)

# Strict warnings for project-owned code (not propagated to ggml)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kokoro PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
    if(KOKORO_WERROR)
        target_compile_options(kokoro PRIVATE -Werror)
    endif()
endif()

target_compile_definitions(kokoro PRIVATE
    KOKORO_BUILD
    KOKORO_VERSION="${PROJECT_VERSION}"
    KOKORO_GIT_HASH="${KOKORO_GIT_HASH}"
)
if(KOKORO_HAS_VULKAN)
    target_compile_definitions(kokoro PRIVATE KOKORO_HAS_VULKAN)
endif()
if(KOKORO_HAS_METAL)
    target_compile_definitions(kokoro PRIVATE KOKORO_HAS_METAL)
endif()
if(KOKORO_HAS_CUDA)
    target_compile_definitions(kokoro PRIVATE KOKORO_HAS_CUDA)
endif()

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

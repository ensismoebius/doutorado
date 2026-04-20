#+#+#+#+-----------------------------------------------------------------------
# Tooling.cmake
#
# Centralized “developer experience” tooling:
# - Detect optional tools (ccache, cppcheck, flawfinder, clang-tidy, valgrind).
# - Enable `compile_commands.json` export for clangd/IDE integration.
# - Wire clang-tidy into compilation when available.
#
# This module is intentionally side-effectful (sets CMake variables / launchers).
#+#+#+#+-----------------------------------------------------------------------

# CMake/Tooling.cmake

# Performance-oriented toggles for local development builds.
option(NN_ENABLE_CCACHE "Enable ccache compiler launcher when available" ON)
option(NN_ENABLE_CLANG_TIDY "Enable clang-tidy during compilation (can be much slower)" OFF)

# --------------------------------------------------------------------------------
# CCACHE Support
# --------------------------------------------------------------------------------
# Add this before the project() command for maximum effect
find_program(CCACHE_FOUND ccache)
if(NN_ENABLE_CCACHE)
    if(CCACHE_FOUND)
        message(STATUS "ccache found at ${CCACHE_FOUND}, enabling for C/C++ compilation")
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_FOUND}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_FOUND}")
        message(STATUS "Tip: run 'ccache -s' to verify cache hit rate and size")
    else()
        message(STATUS "ccache requested but not found, proceeding without it")
    endif()
else()
    message(STATUS "ccache disabled (NN_ENABLE_CCACHE=OFF)")
endif()

# --------------------------------------------------------------------------------
# Analysis Tools Find Program
# --------------------------------------------------------------------------------
find_program(CPPCHECK_EXECUTABLE cppcheck)
find_program(FLAWFINDER_EXECUTABLE flawfinder)
find_program(CLANG_TIDY_EXECUTABLE clang-tidy)

# Generate compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------------------------------------------------------------
# Keep .clangd CompilationDatabase aligned with current configure/build directory
# --------------------------------------------------------------------------------
# On every configure pass, rewrite only the `CompilationDatabase:` line in .clangd
# so editor indexing follows the active CMake build directory.
set(NN_CLANGD_CONFIG_FILE "${CMAKE_SOURCE_DIR}/.clangd")
if(EXISTS "${NN_CLANGD_CONFIG_FILE}")
    file(RELATIVE_PATH NN_CLANGD_COMPDB_REL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")
    if(NN_CLANGD_COMPDB_REL STREQUAL "")
        set(NN_CLANGD_COMPDB_REL ".")
    endif()

    file(READ "${NN_CLANGD_CONFIG_FILE}" NN_CLANGD_CONFIG_CONTENT)
    string(REGEX MATCH "[ \t]*CompilationDatabase:[ \t]*[^\r\n]*" NN_CLANGD_COMPDB_LINE "${NN_CLANGD_CONFIG_CONTENT}")

    if(NN_CLANGD_COMPDB_LINE)
        string(
            REGEX REPLACE
                "([ \t]*CompilationDatabase:[ \t]*)[^\r\n]*"
                "\\1${NN_CLANGD_COMPDB_REL}"
                NN_CLANGD_CONFIG_UPDATED
                "${NN_CLANGD_CONFIG_CONTENT}")

        if(NOT NN_CLANGD_CONFIG_UPDATED STREQUAL NN_CLANGD_CONFIG_CONTENT)
            file(WRITE "${NN_CLANGD_CONFIG_FILE}" "${NN_CLANGD_CONFIG_UPDATED}")
            message(STATUS ".clangd: CompilationDatabase -> ${NN_CLANGD_COMPDB_REL}")
        endif()
    else()
        message(STATUS ".clangd found but no CompilationDatabase line to update")
    endif()
endif()

# --------------------------------------------------------------------------------
# Clang-Tidy Integration
# --------------------------------------------------------------------------------
if(NN_ENABLE_CLANG_TIDY)
    if(CLANG_TIDY_EXECUTABLE)
        message(STATUS "clang-tidy found at ${CLANG_TIDY_EXECUTABLE}, enabling for C++ compilation")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE};--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy")
    else()
        message(STATUS "clang-tidy requested but not found, proceeding without it")
    endif()
else()
    message(STATUS "clang-tidy disabled (NN_ENABLE_CLANG_TIDY=OFF) for faster incremental builds")
endif()

# --------------------------------------------------------------------------------
# Valgrind Callgrind Support for Performance Profiling
# --------------------------------------------------------------------------------
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    message(STATUS "valgrind found, Callgrind available for performance profiling")
else()
    message(STATUS "valgrind not found, Callgrind profiling not available")
endif()

# Function to add a Callgrind profiling target for an executable.
# When valgrind is unavailable, provide a no-op target so configure still succeeds.
function(add_callgrind_target target_name executable)
    if(VALGRIND_EXECUTABLE)
        add_custom_target(${target_name}
            COMMAND ${VALGRIND_EXECUTABLE} --tool=callgrind --callgrind-out-file=callgrind.out.${target_name} $<TARGET_FILE:${executable}>
            DEPENDS ${executable}
            COMMENT "Running ${executable} with Callgrind profiler (output: callgrind.out.${target_name})"
        )
    else()
        add_custom_target(${target_name}
            COMMAND ${CMAKE_COMMAND} -E echo "Skipping ${target_name}: valgrind not found in PATH"
            DEPENDS ${executable}
            COMMENT "Valgrind unavailable, creating no-op Callgrind target ${target_name}"
        )
    endif()
endfunction()

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

# --------------------------------------------------------------------------------
# CCACHE Support
# --------------------------------------------------------------------------------
# Add this before the project() command for maximum effect
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "ccache found, enabling for C/C++ compilation")
    set(CMAKE_C_COMPILER_LAUNCHER ccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
else()
    message(STATUS "ccache not found, proceeding without it")
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
# Clang-Tidy Integration
# --------------------------------------------------------------------------------
if(CLANG_TIDY_EXECUTABLE)
    message(STATUS "clang-tidy found, enabling for C++ compilation")
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE};--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy")
else()
    message(STATUS "clang-tidy not found, proceeding without it")
endif()

# --------------------------------------------------------------------------------
# Valgrind Callgrind Support for Performance Profiling
# --------------------------------------------------------------------------------
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    message(STATUS "valgrind found, Callgrind available for performance profiling")

    # Function to add Callgrind profiling target for an executable
    function(add_callgrind_target target_name executable)
        add_custom_target(${target_name}
            COMMAND ${VALGRIND_EXECUTABLE} --tool=callgrind --callgrind-out-file=callgrind.out.${target_name} $<TARGET_FILE:${executable}>
            DEPENDS ${executable}
            COMMENT "Running ${executable} with Callgrind profiler (output: callgrind.out.${target_name})"
        )
    endfunction()
else()
    message(STATUS "valgrind not found, Callgrind profiling not available")
endif()

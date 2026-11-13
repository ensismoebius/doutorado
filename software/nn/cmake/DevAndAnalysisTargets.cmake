#+#+#+#+-----------------------------------------------------------------------
# DevAndAnalysisTargets.cmake
#
# Adds convenience targets for local development and code-quality checks.
# Typical usage:
# - `cmake --build build --target dev-setup`
 # - `cmake --build build --target analysis-all`
#
# Notes:
# - analysis-cppcheck / -flawfinder / -clang-tidy exist when their tool is found;
#   analysis-all always exists and fails when any of the three is missing.
# - They should never affect normal builds of the library/executables.
#+#+#+#+-----------------------------------------------------------------------

# CMake/DevAndAnalysisTargets.cmake

# --------------------------------------------------------------------------------
# Custom Targets for Development and Analysis
# --------------------------------------------------------------------------------

find_package(Python3 COMPONENTS Interpreter)
find_program(LCOV_EXECUTABLE lcov)

if(CCACHE_FOUND)
    add_custom_target(clean-cache
        COMMAND ${CCACHE_FOUND} -C
        COMMENT "Clearing ccache compilation cache"
        USES_TERMINAL
    )
endif()

# Target to help developers set up required tools
add_custom_target(dev-setup
    COMMENT "Checking for required developer tools..."
)

if(NOT CPPCHECK_EXECUTABLE)
    add_custom_command(TARGET dev-setup POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Warning: 'cppcheck' is not installed. Please install it for static analysis."
    )
endif()

if(NOT FLAWFINDER_EXECUTABLE)
    add_custom_command(TARGET dev-setup POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Warning: 'flawfinder' is not installed. Please install it for security analysis."
    )
endif()

if(NOT CLANG_TIDY_EXECUTABLE)
    add_custom_command(TARGET dev-setup POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Warning: 'clang-tidy' is not installed. Please install it for static analysis."
    )
endif()


if(CPPCHECK_EXECUTABLE)
    include(ProcessorCount)
    ProcessorCount(NN_CPPCHECK_JOBS)
    if(NN_CPPCHECK_JOBS EQUAL 0)
        message(FATAL_ERROR "Cannot determine the processor count for cppcheck -j")
    endif()
    # cppcheck reads the compile database, so every translation unit is analysed with its real
    # include paths and definitions (a bare glob of files, as before, analysed each header on
    # its own and reported members used elsewhere as unused). Only project code is reported:
    # build-tree and vendored (_deps) headers are suppressed. Any finding fails the target, so
    # the code is kept at zero; a justified exception is an inline
    # `// cppcheck-suppress <id>` with its reason. Two checks are off on purpose, as a
    # decision about the whole code base (the same status as a check left out of .clang-tidy):
    #   useStlAlgorithm - "this loop could be std::transform/accumulate": a style preference,
    #                     and the loops it names (138 when measured) are clearer as loops;
    #   shadowFunction  - a local named like a member function (`size`, `rows`): legal,
    #                     unambiguous at the use site, and renaming 52 locals changes nothing.
    add_custom_target(analysis-cppcheck
        COMMAND ${CPPCHECK_EXECUTABLE}
            --project=${CMAKE_BINARY_DIR}/compile_commands.json
            --enable=warning,style,performance,portability
            --inline-suppr
            --library=googletest
            --suppress=missingIncludeSystem
            --suppress=unmatchedSuppression
            --suppress=useStlAlgorithm
            --suppress=shadowFunction
            --suppress=*:*/_deps/*
            --suppress=*:${CMAKE_BINARY_DIR}/*
            -i${CMAKE_BINARY_DIR}
            --std=c++20
            -j ${NN_CPPCHECK_JOBS}
            --xml
            --output-file=${CMAKE_BINARY_DIR}/cppcheck-report.xml
            --error-exitcode=1
        COMMENT "Running cppcheck on the compile database (fails on any finding)..."
        USES_TERMINAL
    )
endif()

if(FLAWFINDER_EXECUTABLE)
    # Find all source and header files for flawfinder
    file(GLOB_RECURSE FLAWFINDER_SOURCES
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.c"
        "${CMAKE_SOURCE_DIR}/src/*.hpp"
        "${CMAKE_SOURCE_DIR}/src/*.h"
    )
    list(FILTER FLAWFINDER_SOURCES EXCLUDE REGEX "build/_deps")

    add_custom_target(analysis-flawfinder
        COMMAND ${FLAWFINDER_EXECUTABLE}
            --minlevel=1
            --html
            --neverignore
            --error-level=5 # Exit on high-risk issues (level 5)
            ${FLAWFINDER_SOURCES} > ${CMAKE_BINARY_DIR}/flawfinder-report.html
        COMMENT "Running flawfinder security analysis..."
        USES_TERMINAL
    )
endif()

if(CLANG_TIDY_EXECUTABLE)
    # Find all source files for clang-tidy
    file(GLOB_RECURSE CLANG_TIDY_SOURCES
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.c"
    )
    list(FILTER CLANG_TIDY_SOURCES EXCLUDE REGEX "build/_deps")

    add_custom_target(analysis-clang-tidy
        COMMAND ${CLANG_TIDY_EXECUTABLE}
            --config-file=${CMAKE_SOURCE_DIR}/.clang-tidy
            -p ${CMAKE_BINARY_DIR}
            ${CLANG_TIDY_SOURCES}
        COMMENT "Running clang-tidy static analysis..."
        USES_TERMINAL
    )
endif()


# analysis-all runs every analyser. A missing one makes the target FAIL with its name; it used
# to fall back silently to the tools that were installed (no clang-tidy: no clang-tidy run,
# and "analysis-all" still passed).
add_custom_target(analysis-all
    COMMENT "Running all analysis tools (cppcheck, flawfinder, clang-tidy)..."
)
set(_nn_missing_analysers "")
foreach(_tool cppcheck flawfinder clang-tidy)
    string(TOUPPER "${_tool}" _var)
    string(REPLACE "-" "_" _var "${_var}")
    if(${_var}_EXECUTABLE)
        add_dependencies(analysis-all analysis-${_tool})
    else()
        list(APPEND _nn_missing_analysers ${_tool})
    endif()
endforeach()
if(_nn_missing_analysers)
    add_custom_command(TARGET analysis-all PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "analysis-all: not installed: ${_nn_missing_analysers}"
        COMMAND ${CMAKE_COMMAND} -E false
    )
endif()

if(NN_ENABLE_COVERAGE AND Python3_Interpreter_FOUND AND LCOV_EXECUTABLE)
    # Script path is scripts/dev/, NOT tools/ -- this pointed at a nonexistent
    # ${CMAKE_SOURCE_DIR}/tools/check_core_coverage.py, so the target existed but could
    # never run (it failed with "can't open file" the moment anyone invoked it). Guarded
    # below so a future move breaks at configure time instead of at invocation time.
    set(_nn_coverage_script ${CMAKE_SOURCE_DIR}/scripts/dev/check_core_coverage.py)
    if(NOT EXISTS ${_nn_coverage_script})
        message(FATAL_ERROR
            "NN_ENABLE_COVERAGE=ON but the coverage gate script is missing: ${_nn_coverage_script}")
    endif()
    add_custom_target(coverage-core-100
        COMMAND ${Python3_EXECUTABLE}
            ${_nn_coverage_script}
            --build-dir ${CMAKE_BINARY_DIR}
            --line-threshold 100
            --function-threshold 100
        COMMENT "Running strict 100% coverage gate for core libraries"
        USES_TERMINAL
    )
endif()



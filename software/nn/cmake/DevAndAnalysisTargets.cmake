#+#+#+#+-----------------------------------------------------------------------
# DevAndAnalysisTargets.cmake
#
# Adds convenience targets for local development and code-quality checks.
# Typical usage:
# - `cmake --build build --target dev-setup`
# - `cmake --build build --target analysis-all`
# - `cmake --build build --target check_eigen_leaks`
#
# Notes:
# - These targets are *optional* (they exist only when tools are found).
# - They should never affect normal builds of the library/executables.
#+#+#+#+-----------------------------------------------------------------------

# CMake/DevAndAnalysisTargets.cmake

# --------------------------------------------------------------------------------
# Custom Targets for Development and Analysis
# --------------------------------------------------------------------------------

find_package(Python3 COMPONENTS Interpreter)

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
    # Define SRC_DIR if not already defined (assuming project's main src directory)
    if(NOT DEFINED SRC_DIR)
        set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
    endif()

    file(GLOB_RECURSE CPPCHECK_FILES
        "${SRC_DIR}/*.cpp"
        "${SRC_DIR}/*.c"
        "${SRC_DIR}/*.hpp"
        "${SRC_DIR}/*.h"
    )
    # Filter out files from _deps directories that might have been included
    list(FILTER CPPCHECK_FILES EXCLUDE REGEX "build/_deps")

    add_custom_target(analysis-cppcheck
        COMMAND ${CPPCHECK_EXECUTABLE}
            ${CPPCHECK_FILES}
            --enable=warning,style,performance,portability,information
            --suppress=missingIncludeSystem
            --suppress=unmatchedSuppression
            --suppress=syntaxError
            --suppress=internalAstError
            --suppress=containerOutOfBounds
            --std=c++20
            --cpp-header-probe
            --xml
            --output-file=${CMAKE_BINARY_DIR}/cppcheck-report.xml
            --error-exitcode=0
        COMMENT "Running cppcheck static analysis on project sources..."
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


if(CPPCHECK_EXECUTABLE AND FLAWFINDER_EXECUTABLE AND CLANG_TIDY_EXECUTABLE)
    add_custom_target(analysis-all
        COMMENT "Running all analysis tools (cppcheck, flawfinder, clang-tidy)..."
    )
    # Using DEPENDS ensures that both targets are triggered.
    # If one fails, the overall build command will fail, but CMake
    # may still attempt to run the other target in parallel.
    add_dependencies(analysis-all analysis-cppcheck analysis-flawfinder analysis-clang-tidy)
elseif(CPPCHECK_EXECUTABLE AND FLAWFINDER_EXECUTABLE)
    add_custom_target(analysis-all
        COMMENT "Running all analysis tools (cppcheck, flawfinder)..."
    )
    add_dependencies(analysis-all analysis-cppcheck analysis-flawfinder)
endif()

if(Python3_Interpreter_FOUND)
    # Eigen leak detection (static check)
    add_custom_target(check_eigen_leaks
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_SOURCE_DIR}/scripts
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/check_eigen_leaks.py
                --allowlist ${CMAKE_SOURCE_DIR}/eigen_allowlist.txt
                --root ${CMAKE_SOURCE_DIR}
                --paths src include
        COMMENT "Detecting Eigen usage outside the allowlist"
        USES_TERMINAL
    )
else()
    message(WARNING "Python3 interpreter not found; 'check_eigen_leaks' target disabled.")
endif()

# CMake/DevAndAnalysisTargets.cmake

# --------------------------------------------------------------------------------
# Custom Targets for Development and Analysis
# --------------------------------------------------------------------------------

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


if(CPPCHECK_EXECUTABLE)
    add_custom_target(analysis-cppcheck
        COMMAND ${CPPCHECK_EXECUTABLE}
            --project=${CMAKE_BINARY_DIR}/compile_commands.json
            --enable=warning,style,performance,portability,information
            --suppress=missingIncludeSystem
            --suppress=unmatchedSuppression
            --std=c++20
            --xml
            --output-file=${CMAKE_BINARY_DIR}/cppcheck-report.xml
            --error-exitcode=1
        COMMENT "Running cppcheck static analysis..."
        USES_TERMINAL
    )
endif()

if(FLAWFINDER_EXECUTABLE)
    # Find all source and header files for flawfinder
    file(GLOB_RECURSE FLAWFINDER_SOURCES
        "${SRC_DIR}/*.cpp"
        "${SRC_DIR}/*.c"
        "${SRC_DIR}/*.hpp"
        "${SRC_DIR}/*.h"
    )

    add_custom_target(analysis-flawfinder
        COMMAND ${FLAWFINDER_EXECUTABLE}
            --minlevel=1
            --html
            --never-ignore
            --exit-on-risk=5 # Exit on high-risk issues (level 5)
            ${FLAWFINDER_SOURCES} > ${CMAKE_BINARY_DIR}/flawfinder-report.html
        COMMENT "Running flawfinder security analysis..."
        USES_TERMINAL
    )
endif()


if(CPPCHECK_EXECUTABLE AND FLAWFINDER_EXECUTABLE)
    add_custom_target(analysis-all
        COMMENT "Running all analysis tools (cppcheck, flawfinder)..."
    )
    # Using DEPENDS ensures that both targets are triggered.
    # If one fails, the overall build command will fail, but CMake
    # may still attempt to run the other target in parallel.
    add_dependencies(analysis-all analysis-cppcheck analysis-flawfinder)
endif()

# Gtest and Gmock configuration
set(GTEST_DIR "${LIB_DIR}/gtest")
set(GTEST_BUILD_DIR "${GTEST_DIR}/build")

# Check if Gtest and Gmock is already downloaded and compiled
if(NOT EXISTS "${GTEST_BUILD_DIR}/lib/libgtest.a" OR NOT EXISTS "${GTEST_DIR}/googletest/include/gtest/gtest.h")
    message(STATUS "Gtest and Gmock not found, downloading and compiling...")

    if(NOT EXISTS "${GTEST_DIR}/.git")
        # Clone Gtest and Gmock repository if not already cloned
        execute_process(
            COMMAND git clone --depth 1 https://github.com/google/googletest ${GTEST_DIR}
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to clone Gtest and Gmock repository")
        endif()
    else()
        # Update Gtest and Gmock repository if already cloned
        execute_process(
            COMMAND git -C ${GTEST_DIR} pull
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to update Gtest and Gmock repository")
        endif()
    endif()

    # Create build directory and compile Gtest and Gmock with -fPIC
    file(MAKE_DIRECTORY ${GTEST_BUILD_DIR})
    execute_process(
        COMMAND cmake -S ${GTEST_DIR} -B ${GTEST_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        RESULT_VARIABLE result
    )
    if(result)
        message(FATAL_ERROR "CMake configuration for Gtest and Gmock failed")
    endif()

    execute_process(
        COMMAND cmake --build ${GTEST_BUILD_DIR} --parallel
        RESULT_VARIABLE result
    )
    if(result)
        message(FATAL_ERROR "Gtest and Gmock build failed")
    endif()
else()
    message(STATUS "Gtest and Gmock already exists, skipping download and compilation...")
endif()

# Define Gtest and Gmock library
add_library(gtest STATIC IMPORTED)
set_target_properties(gtest PROPERTIES
    IMPORTED_LOCATION "${GTEST_BUILD_DIR}/lib/libgtest.a"
    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_DIR}/googletest/include"
)

# Define Gmock library
add_library(gmock STATIC IMPORTED)
set_target_properties(gmock PROPERTIES
    IMPORTED_LOCATION "${GTEST_BUILD_DIR}/lib/libgmock.a"
    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_DIR}/googlemock/include"
)
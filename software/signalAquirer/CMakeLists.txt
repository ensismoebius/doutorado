cmake_minimum_required(VERSION 3.10)
project(SignalAcquirer VERSION 1.0 LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define paths
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
set(RAYLIB_DIR "${LIB_DIR}/raylib")
set(RAYLIB_BUILD_DIR "${RAYLIB_DIR}/build")

# Add Armadillo and OpenMP dependencies
find_package(Armadillo REQUIRED)
find_package(OpenMP REQUIRED)

# Find ALSA library
find_package(PkgConfig REQUIRED)
pkg_check_modules(ASOUND REQUIRED alsa)

# Check if raylib is already downloaded and compiled
if(NOT EXISTS "${RAYLIB_BUILD_DIR}/libraylib.a")
    message(STATUS "Raylib not found, downloading and compiling...")

    if(NOT EXISTS "${RAYLIB_DIR}/.git")
        # Clone raylib repository if not already cloned
        execute_process(
            COMMAND git clone https://github.com/raysan5/raylib ${RAYLIB_DIR}
            RESULT_VARIABLE result
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(result)
            message(FATAL_ERROR "Failed to clone raylib repository")
        endif()
    else()
        # Update raylib repository if already cloned
        execute_process(
            COMMAND git -C ${RAYLIB_DIR} pull
            RESULT_VARIABLE result
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(result)
            message(FATAL_ERROR "Failed to update raylib repository")
        endif()
    endif()

    # Create build directory and compile raylib
    file(MAKE_DIRECTORY ${RAYLIB_BUILD_DIR})
    execute_process(
        COMMAND cmake -S ${RAYLIB_DIR} -B ${RAYLIB_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release
        RESULT_VARIABLE result
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(result)
        message(FATAL_ERROR "CMake configuration for raylib failed")
    endif()

    execute_process(
        COMMAND cmake --build ${RAYLIB_BUILD_DIR} --parallel
        RESULT_VARIABLE result
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(result)
        message(FATAL_ERROR "Raylib build failed")
    endif()
else()
    message(STATUS "Raylib already exists, skipping download and compilation...")
endif()

# Define raylib library
add_library(raylib STATIC IMPORTED)
set_target_properties(raylib PROPERTIES
    IMPORTED_LOCATION "${RAYLIB_BUILD_DIR}/raylib/libraylib.a"
    INTERFACE_INCLUDE_DIRECTORIES "${RAYLIB_DIR}/src"
)

# Add executable target
add_executable(SignalAcquirer ${SRC_DIR}/main.cpp)
target_link_libraries(SignalAcquirer
    PRIVATE
        ${ARMADILLO_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
        raylib
        ${ASOUND_LIBRARIES}
)
target_include_directories(SignalAcquirer
    PRIVATE
        ${SRC_DIR}
        ${ARMADILLO_INCLUDE_DIRS}
        ${OpenMP_INCLUDE_DIRS}
        ${ASOUND_INCLUDE_DIRS}
)

# Print configuration for debugging
message(STATUS "ASOUND_INCLUDE_DIRS: ${ASOUND_INCLUDE_DIRS}")
message(STATUS "ASOUND_LIBRARIES: ${ASOUND_LIBRARIES}")

cmake_minimum_required(VERSION 3.10)
project(nn VERSION 0.1.0 LANGUAGES CXX)

# Sets up the flags for CMake and compiler
include(Flags.cmake)

# Do the required libraries check
include(PackageChecking.cmake)

# Define paths needed for the step bellow
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")

include(cnpy.cmake)

add_subdirectory(${CNPY_DIR})

# Add the executable
add_executable(nn
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/layers/ReLU.cpp
    ${SRC_DIR}/layers/Linear.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/util/vetorizationCheck.cpp
)

# Link libraries
target_link_libraries(nn
    PRIVATE
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
)

# Include directories
target_include_directories(nn
    PRIVATE
        ${SRC_DIR}
        ${CNPY_DIR}
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
)
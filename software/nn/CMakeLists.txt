cmake_minimum_required(VERSION 3.5.0)
project(nn VERSION 0.1.0 LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set debug flags for the Debug build type
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -ggdb -O0 -fno-inline -fno-omit-frame-pointer")

# Set optimization flags for the Release build type
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")

# Add the executable
add_executable(
    nn
    main.cpp
    layers/ReLU.cpp
    layers/Linear.cpp
    util/vetorizationCheck.cpp
)

# Find Eigen (you may need to adjust the path to where Eigen is installed)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

# Include Eigen directories
target_link_libraries(nn Eigen3::Eigen)

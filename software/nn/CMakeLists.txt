cmake_minimum_required(VERSION 3.5.0)
project(nn VERSION 0.1.0 LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(
    nn
    main.cpp
    Relu.cpp
    Layer.cpp
    Linear.cpp
    Model.cpp
)

# Find Eigen (you may need to adjust the path to where Eigen is installed)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include Eigen directories
target_link_libraries(nn Eigen3::Eigen)

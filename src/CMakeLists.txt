cmake_minimum_required(VERSION 3.0.0)
project(Doutorado VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(Doutorado 
main.cpp
lib/neuralNetwork/Neuralnet.cpp
lib/neuralNetwork/layer/Layer.cpp
lib/neuralNetwork/optimizer/Optimizer.cpp
lib/neuralNetwork/output/Output.cpp
lib/neuralNetwork/utils/Rng.cpp
)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (Doutorado Eigen3::Eigen)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

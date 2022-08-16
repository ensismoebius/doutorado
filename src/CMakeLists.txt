cmake_minimum_required(VERSION 3.0.0)
project(Doutorado VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(Doutorado 
main.cpp
lib/neuralNetwork/optimizer/AdamOptimizer.cpp
lib/neuralNetwork/model/Sequential.cpp
lib/neuralNetwork/layer/Dense.cpp
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

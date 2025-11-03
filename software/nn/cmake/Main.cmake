add_subdirectory(${CNPY_DIR})

# Add subdirectories for all testable modules
add_subdirectory(${SRC_DIR}/util)
add_subdirectory(${SRC_DIR}/layers)
add_subdirectory(${SRC_DIR}/tensor)
add_subdirectory(${SRC_DIR}/optimizers)
add_subdirectory(${SRC_DIR}/initializers)
add_subdirectory(${SRC_DIR}/dataLoaders)

# Include modular CMake target files
include(${CMAKE_SOURCE_DIR}/cmake/exec_AutoEncoderTargets.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_plotSpikingNetwork.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_mainProject.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_resnet_demo.cmake)
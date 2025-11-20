# Include all demo subdirectories
add_subdirectory(${SRC_DIR}/core/wave/demo)

# Include modular CMake target files
include(${CMAKE_SOURCE_DIR}/cmake/exec_loadingData.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_mainProject.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_resnet_demo.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_Experiment01.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_AutoEncoderTargets.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_plotSpikingNetwork.cmake)
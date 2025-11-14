

# Add subdirectories for all testable modules
add_subdirectory(${SRC_DIR}/util)
add_subdirectory(${SRC_DIR}/core)
add_subdirectory(${SRC_DIR}/core/layers)
add_subdirectory(${SRC_DIR}/core/layers/tests)
add_subdirectory(${SRC_DIR}/core/tensor)
add_subdirectory(${SRC_DIR}/core/tensor/tests)
add_subdirectory(${SRC_DIR}/core/optimizers)
add_subdirectory(${SRC_DIR}/core/optimizers/tests)
add_subdirectory(${SRC_DIR}/core/initializers)
add_subdirectory(${SRC_DIR}/core/initializers/tests)
add_subdirectory(${SRC_DIR}/core/dataLoaders)
add_subdirectory(${SRC_DIR}/core/dataLoaders/tests)
add_subdirectory(${SRC_DIR}/core/utility/tests)
add_subdirectory(${SRC_DIR}/core/linearAlgebra)
add_subdirectory(${SRC_DIR}/core/linearAlgebra/tests)
add_subdirectory(${SRC_DIR}/core/wave)
add_subdirectory(${SRC_DIR}/core/wave/tests)
add_subdirectory(${SRC_DIR}/core/wavelet)
add_subdirectory(${SRC_DIR}/core/wavelet/tests)
add_subdirectory(${SRC_DIR}/core/utility)
add_subdirectory(${SRC_DIR}/core/paraconsistent)
add_subdirectory(${SRC_DIR}/core/paraconsistent/tests)
add_subdirectory(${SRC_DIR}/core/statistics)
add_subdirectory(${SRC_DIR}/core/statistics/tests)

# Include all demo subdirectories
add_subdirectory(${SRC_DIR}/core/wave/demo)

# Include modular CMake target files
include(${CMAKE_SOURCE_DIR}/cmake/exec_AutoEncoderTargets.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_plotSpikingNetwork.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_mainProject.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_resnet_demo.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_Experiment01.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/exec_loadingData.cmake)
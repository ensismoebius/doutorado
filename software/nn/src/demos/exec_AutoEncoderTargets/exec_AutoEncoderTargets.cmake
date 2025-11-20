## AutoEncoderTargets.cmake
# Targets for auto-encoder tests

add_executable(autoEncoderLeakyReLUAndSpikeTest
    ${SRC_DIR}/util/synthetic_spike_data.cpp
    ${SRC_DIR}/util/vectorizationCheck.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/core/NnSaver.hpp
    ${SRC_DIR}/experiments/autoEncoderLeakyReLUAndSpikeTest.cpp
)

# Link libraries
target_link_libraries(autoEncoderLeakyReLUAndSpikeTest
    PRIVATE
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
)

# Include directories
target_include_directories(autoEncoderLeakyReLUAndSpikeTest
    PRIVATE
        ${SRC_DIR}
        ${SRC_DIR}/core/initializers
        ${SRC_DIR}/core/optimizers
        ${SRC_DIR}/core/layers
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
        ${cnpy_SOURCE_DIR}
)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(autoEncoderLeakyReLUAndSpikeTest)


add_executable(autoEncoderLeakyReLUTest
    ${SRC_DIR}/util/synthetic_spike_data.cpp
    ${SRC_DIR}/util/vectorizationCheck.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/core/NnSaver.hpp
    ${SRC_DIR}/experiments/autoEncoderLeakyReLUTest.cpp
)

# Link libraries
target_link_libraries(autoEncoderLeakyReLUTest
    PRIVATE
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
)

# Include directories
target_include_directories(autoEncoderLeakyReLUTest
    PRIVATE
        ${SRC_DIR}
        ${SRC_DIR}/core/initializers
        ${SRC_DIR}/core/optimizers
        ${SRC_DIR}/core/layers
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
        ${cnpy_SOURCE_DIR}
)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(autoEncoderLeakyReLUTest)

add_executable(autoEncoderLeakyReLUAndSpikeTest
    ../autoEncoderLeakyReLUAndSpikeTest/autoEncoderLeakyReLUAndSpikeTest.cpp
)

# Link libraries
target_link_libraries(autoEncoderLeakyReLUAndSpikeTest
    PRIVATE
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
        util
        tensor
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
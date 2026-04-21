# Combined autoencoder LeakyReLU tests

# Test 1: autoencoder_scaffold_example
add_executable(autoencoder_scaffold_test
    autoencoder_scaffold_example.cpp
)
target_link_libraries(autoencoder_scaffold_test
    PRIVATE
        experiment03_lib
        Eigen3::Eigen
)
target_include_directories(autoencoder_scaffold_test
    PRIVATE
        ${SRC_DIR}
        ${SRC_DIR}/experiments/03/lib/include
        ${SRC_DIR}/experiments/03/lib/include/autoencoder
        ${PROJECT_SOURCE_DIR}/include
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
        ${cnpy_SOURCE_DIR}
)
configure_eigen_parallel_target(autoencoder_scaffold_test)

# Test 2: autoEncoderLeakyReLUTest
add_executable(autoEncoderLeakyReLUTest
    autoEncoderLeakyReLUTest.cpp
)
target_link_libraries(autoEncoderLeakyReLUTest
    PRIVATE
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
        util
        tensor
        layers
        optimizers
)
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
configure_eigen_parallel_target(autoEncoderLeakyReLUTest)

# Test 3: autoEncoderLeakyReLUAndSpikeTest
add_executable(autoEncoderLeakyReLUAndSpikeTest
    autoEncoderLeakyReLUAndSpikeTest.cpp
)
target_link_libraries(autoEncoderLeakyReLUAndSpikeTest
    PRIVATE
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
        util
        tensor
        layers
        initializers
        optimizers
)
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
configure_eigen_parallel_target(autoEncoderLeakyReLUAndSpikeTest)
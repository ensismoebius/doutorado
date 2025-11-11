## AutoEncoderTargets.cmake
# Targets for auto-encoder tests

add_executable(autoEncoderLeakyReLUAndSpikeTest
    ${CNPY_SOURCES}
    ${SRC_DIR}/initializers/xavier.hpp
    ${SRC_DIR}/optimizers/SGDMinimal.hpp
    ${SRC_DIR}/optimizers/Adam.hpp
    ${SRC_DIR}/optimizers/SGD.hpp
    ${SRC_DIR}/layers/Sequential.hpp
    ${SRC_DIR}/layers/Linear.hpp
    ${SRC_DIR}/layers/Leaky.hpp
    ${SRC_DIR}/layers/ReLU.hpp
    ${SRC_DIR}/layers/LeakyReLU.hpp
    ${SRC_DIR}/util/synthetic_spike_data.cpp
    ${SRC_DIR}/util/vectorizationCheck.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/util/NnSaver.hpp
    ${SRC_DIR}/experiments/autoEncoderLeakyReLUAndSpikeTest.cpp
)

# Link libraries
target_link_libraries(autoEncoderLeakyReLUAndSpikeTest
    PUBLIC
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
)

# Include directories
target_include_directories(autoEncoderLeakyReLUAndSpikeTest
    PUBLIC
        ${SRC_DIR}
        "${cnpy_SOURCE_DIR}"
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(autoEncoderLeakyReLUAndSpikeTest)


add_executable(autoEncoderLeakyReLUTest
    ${CNPY_SOURCES}
    ${SRC_DIR}/initializers/xavier.hpp
    ${SRC_DIR}/optimizers/SGDMinimal.hpp
    ${SRC_DIR}/optimizers/Adam.hpp
    ${SRC_DIR}/optimizers/SGD.hpp
    ${SRC_DIR}/layers/Sequential.hpp
    ${SRC_DIR}/layers/Linear.hpp
    ${SRC_DIR}/layers/Leaky.hpp
    ${SRC_DIR}/layers/ReLU.hpp
    ${SRC_DIR}/layers/LeakyReLU.hpp
    ${SRC_DIR}/util/synthetic_spike_data.cpp
    ${SRC_DIR}/util/vectorizationCheck.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/util/NnSaver.hpp
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
        "${cnpy_SOURCE_DIR}"
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(autoEncoderLeakyReLUTest)

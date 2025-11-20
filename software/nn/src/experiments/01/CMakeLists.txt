find_package(ZLIB REQUIRED)

# Experiment 01 CMake configuration
add_executable(exec_Experiment01
    ${SRC_DIR}/core/initializers/xavier.hpp
    ${SRC_DIR}/core/optimizers/SGDMinimal.hpp
    ${SRC_DIR}/core/optimizers/Adam.hpp
    ${SRC_DIR}/core/optimizers/SGD.hpp
    ${SRC_DIR}/core/layers/Sequential.hpp
    ${SRC_DIR}/core/layers/Linear.hpp
    ${SRC_DIR}/core/layers/ReLU.hpp
    ${SRC_DIR}/core/layers/LeakyReLU.hpp
    ${SRC_DIR}/util/synthetic_spike_data.cpp
    ${SRC_DIR}/util/vectorizationCheck.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/core/NnSaver.hpp
    ${SRC_DIR}/core/dataLoaders/10.1117/AudioLoader.cpp
    ${SRC_DIR}/core/dataLoaders/10.1117/EEGLoader.cpp
    ${SRC_DIR}/core/dataLoaders/MatFile.cpp
    ${SRC_DIR}/core/wave/Wav.cpp
    ${SRC_DIR}/experiments/01/Experiment01.cpp
    ${SRC_DIR}/experiments/01/Experiment01_utils.cpp
)

# Include directories
target_include_directories(exec_Experiment01
    PRIVATE
        ${SRC_DIR}
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
        ${cnpy_SOURCE_DIR} # Added back for NetworkSerializer.hpp
)

# Link libraries
target_link_libraries(exec_Experiment01
    PRIVATE
        cnpy
        Eigen3::Eigen
		${ZLIB_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
        matioCpp
        fftw3f
)

add_subdirectory(${SRC_DIR}/tests)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(exec_Experiment01)
# End of Experiment 01 CMake configuration
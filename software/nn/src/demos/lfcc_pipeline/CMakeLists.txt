find_package(ZLIB REQUIRED)

# Experiment 01 CMake configuration
add_library(lfcc_pipeline_utils_lib STATIC
    ${SRC_DIR}/demos/lfcc_pipeline/Experiment01_utils.cpp
)

target_include_directories(lfcc_pipeline_utils_lib
    PRIVATE
        ${SRC_DIR}
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
        ${cnpy_SOURCE_DIR}
)

target_link_libraries(lfcc_pipeline_utils_lib
    PRIVATE
        cnpy
        Eigen3::Eigen
        ${ZLIB_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
        matioCpp
        fftw3f
        util
        tensor
        dataLoaders_10_1117
        waveCoreLib
)

add_executable(exec_lfcc_pipeline
    ${SRC_DIR}/core/initializers/xavier.hpp
    ${SRC_DIR}/core/optimizers/SGDMinimal.hpp
    ${SRC_DIR}/core/optimizers/Adam.hpp
    ${SRC_DIR}/core/optimizers/SGD.hpp
    ${SRC_DIR}/core/layers/Sequential.hpp
    ${SRC_DIR}/core/layers/Linear.hpp
    ${SRC_DIR}/core/layers/ReLU.hpp
    ${SRC_DIR}/core/layers/LeakyReLU.hpp
    ${SRC_DIR}/core/dataLoaders/MatFile.cpp
    ${SRC_DIR}/core/wave/Wav.cpp
    ${SRC_DIR}/demos/lfcc_pipeline/Experiment01.cpp
)

# Include directories
target_include_directories(exec_lfcc_pipeline
    PRIVATE
        ${SRC_DIR}
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
        ${cnpy_SOURCE_DIR} 
)

# Link libraries
target_link_libraries(exec_lfcc_pipeline
    PRIVATE
        cnpy
        Eigen3::Eigen
		${ZLIB_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
        matioCpp
        fftw3f
        util
        tensor
        lfcc_pipeline_utils_lib
        dataLoaders_10_1117
        waveCoreLib
)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(exec_lfcc_pipeline)
# End of Experiment 01 CMake configuration

add_subdirectory(tests)
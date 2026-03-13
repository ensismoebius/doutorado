find_package(ZLIB REQUIRED)

# Experiment 01 CMake configuration
add_executable(exec_lfcc_pipeline
    ${CMAKE_SOURCE_DIR}/include/nn/initializers/xavier.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/optimizers/SGDMinimal.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/optimizers/Adam.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/optimizers/SGD.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/layers/Sequential.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/layers/Linear.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/layers/ReLU.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/layers/LeakyReLU.hpp
    ${SRC_DIR}/core/dataLoaders/mat_file.cpp
    ${SRC_DIR}/core/wave/Wav.cpp
    ${SRC_DIR}/demos/lfcc_pipeline/lfcc_pipeline.cpp
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
        dataLoaders_10_1117
        waveCoreLib
)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(exec_lfcc_pipeline)
# End of Experiment 01 CMake configuration

add_subdirectory(tests)
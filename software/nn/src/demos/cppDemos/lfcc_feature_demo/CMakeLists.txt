find_package(ZLIB REQUIRED)

# Experiment 01 CMake configuration
add_executable(exec_lfcc_pipeline
    ${CMAKE_SOURCE_DIR}/include/initializers/xavier.hpp
    ${CMAKE_SOURCE_DIR}/include/optimizers/SGDMinimal.hpp
    ${CMAKE_SOURCE_DIR}/include/optimizers/Adam.hpp
    ${CMAKE_SOURCE_DIR}/include/optimizers/SGD.hpp
    ${CMAKE_SOURCE_DIR}/include/layers/base/Sequential.hpp
    ${CMAKE_SOURCE_DIR}/include/layers/dense/Linear.hpp
    ${CMAKE_SOURCE_DIR}/include/layers/activations/ReLU.hpp
    ${CMAKE_SOURCE_DIR}/include/layers/activations/LeakyReLU.hpp
    ${SRC_DIR}/core/data_loaders/io/mat_file.cpp
    ${SRC_DIR}/core/wave/Wav.cpp
    ${SRC_DIR}/demos/cppDemos/lfcc_feature_demo/lfcc_pipeline.cpp
)

# Include directories
target_include_directories(exec_lfcc_pipeline
    PRIVATE
        ${SRC_DIR}
        ${LIB_DIR}/util
        ${OpenMP_INCLUDE_DIRS}
        ${cnpy_SOURCE_DIR} 
)

# Link libraries
target_link_libraries(exec_lfcc_pipeline
    PRIVATE
        cnpy
        xtensor xtensor-blas
		${ZLIB_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
        matioCpp
        FFTW::FFTWF
        util
        tensor
        dataLoaders_10_1117
        waveCoreLib
)

# Configure xtensor parallelism for the target
configure_xtensor_parallel_target(exec_lfcc_pipeline)
# End of Experiment 01 CMake configuration

add_subdirectory(tests)
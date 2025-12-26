add_executable(core_gtest
    NetworkSerializer_gtest.cpp
)

target_link_libraries(core_gtest
    PRIVATE 
        google_test 
        Eigen3::Eigen
        cnpy-static
        z
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${OPENBLAS_LIBRARIES}
        tensor
)

target_include_directories(core_gtest
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/src 
        ${CMAKE_SOURCE_DIR}/src/core/tensor 
        ${EIGEN3_INCLUDE_DIR} 
        ${cnpy_SOURCE_DIR}
)

include(GoogleTest)
gtest_discover_tests(core_gtest)
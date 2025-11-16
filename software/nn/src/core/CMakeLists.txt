add_executable(core_gtest
    tests/NetworkSerializer_gtest.cpp
)

target_link_libraries(core_gtest
    PRIVATE 
        GTest::gtest_main 
        Eigen3::Eigen
        cnpy-static
        z
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${OPENBLAS_LIBRARIES}
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

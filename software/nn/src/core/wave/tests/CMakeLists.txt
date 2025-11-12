add_executable(wave_gtest 
    wave_gtest.cpp
)

target_link_libraries(wave_gtest 
    PRIVATE GTest::gtest_main
    PRIVATE filtersOperationsLibStatic
    PRIVATE linearAlgebra
)

target_include_directories(wave_gtest 
    PRIVATE ${CMAKE_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(wave_gtest)

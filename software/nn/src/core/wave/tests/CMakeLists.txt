add_executable(wave_gtest 
    wave_gtest.cpp
    ../filtersOperations.cpp
    ../simpleSignalOperations.cpp
    ../Wav.cpp
    ../../linearAlgebra/linearAlgebra.cpp
)

target_link_libraries(wave_gtest 
    PRIVATE GTest::gtest_main
)

target_include_directories(wave_gtest 
    PRIVATE ${CMAKE_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(wave_gtest)

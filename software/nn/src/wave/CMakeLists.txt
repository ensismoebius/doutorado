add_library(wave STATIC
    Wav.cpp
    filtersOperations.cpp
    simpleSignalOperations.cpp
)

target_include_directories(wave
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(wave_gtest 
    wave_gtest.cpp
)

target_link_libraries(wave_gtest 
    PRIVATE GTest::gtest_main
    PRIVATE wave
)

target_include_directories(wave_gtest 
    PRIVATE ${CMAKE_SOURCE_DIR}/src
)

add_executable(create_test_wav create_test_wav.cpp)
target_link_libraries(create_test_wav PRIVATE wave)

include(GoogleTest)

gtest_discover_tests(wave_gtest)
configure_eigen_parallel_target(wave_gtest)

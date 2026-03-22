add_executable(prefetch_benchmark
    prefetch_benchmark.cpp
)

target_link_libraries(prefetch_benchmark
    PRIVATE
        dataLoaders
    tensor
    Eigen3::Eigen
)

target_include_directories(prefetch_benchmark
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

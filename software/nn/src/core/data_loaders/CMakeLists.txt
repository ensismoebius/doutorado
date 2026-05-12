find_package(ZLIB REQUIRED)

add_subdirectory(samplers)

add_library(data_loaders
    STATIC
    runtime/DataLoader.cpp
    io/mat_file_utils.cpp
    runtime/BatchPrefetcher.cpp
    sources/SqliteBatchSource.cpp
    runtime/DataLoaderIterator.cpp
    options/SamplerOptionResolution.cpp
)

target_include_directories(data_loaders
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(data_loaders
    PRIVATE
        ${ZLIB_LIBRARIES}
        SQLite3::SQLite3
    cnpy
        dataLoaders_samplers
        xtensor xtensor-blas
    matioCpp
        tensor
)

add_subdirectory(tests)
add_subdirectory(10.1117)

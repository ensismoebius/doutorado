find_package(ZLIB REQUIRED)

add_subdirectory(samplers)

add_library(dataLoaders
    STATIC
        DataLoader.cpp
        mat_file_utils.cpp
        BatchPrefetcher.cpp
        DataLoaderIterator.cpp
        SamplerOptionResolution.cpp
)

target_include_directories(dataLoaders
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(dataLoaders
    PRIVATE
        ${ZLIB_LIBRARIES}
        SQLite3::SQLite3
    cnpy
        dataLoaders_samplers
        Eigen3::Eigen
    matioCpp
        tensor
)

add_subdirectory(tests)
add_subdirectory(10.1117)

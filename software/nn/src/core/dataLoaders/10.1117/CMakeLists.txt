add_library(dataLoaders_10_1117
    STATIC
    dataset_info.cpp
    loaders/AudioLoader.cpp
    loaders/EEGLoader.cpp
    schema/SubjectDiscovery.cpp
    codec/InputModeCodec.cpp
    codec/BatchTargetFormatter.cpp
    protocol/Protocol101117Dataset.cpp
    protocol/SamplePacking.cpp
    protocol/SynchronizedBatchAssembler.cpp
)

target_include_directories(dataLoaders_10_1117
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(dataLoaders_10_1117
    PRIVATE
        Eigen3::Eigen
        ${ZLIB_LIBRARIES}
        matioCpp
        tensor
        dataLoaders
    cnpy
)

nn_enable_target_pch(dataLoaders_10_1117)

add_subdirectory(windowing)
add_subdirectory(tests)

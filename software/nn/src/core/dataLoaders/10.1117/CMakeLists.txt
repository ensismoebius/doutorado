file(GLOB PROTOCOL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/datasets/raw/*.cpp)

add_library(dataLoaders_10_1117
    STATIC
    loaders/AudioLoader.cpp
    loaders/EEGLoader.cpp
    schema/SubjectDiscovery.cpp
    codec/InputModeCodec.cpp
    codec/BatchTargetFormatter.cpp
    ${PROTOCOL_SRCS}
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

# Prefer vendored or system SQLite. Use canonical `SQLite3::SQLite3` alias.
if(NOT TARGET SQLite3::SQLite3)
    message(FATAL_ERROR "Required CMake target 'SQLite3::SQLite3' not found.\n"
            "Either enable vendored SQLite (configure with -DUSE_VENDORED_SQLITE=ON)\n"
            "or install a CMake package that provides the SQLite3::SQLite3 target.")
endif()

target_link_libraries(dataLoaders_10_1117 PRIVATE SQLite3::SQLite3)
 

nn_enable_target_pch(dataLoaders_10_1117)

## dataLoaders_10_1117_windowing — sliding-window datasets for EEG + audio multimodal pipelines.
add_library(dataLoaders_10_1117_windowing
    STATIC
        datasets/windowed/EEGWindowDataset.cpp
        datasets/windowed/AudioWindowDataset.cpp
        datasets/windowed/FusedWindowDataset.cpp
    datasets/windowed/WindowingDatasetPrinter.cpp
)

target_include_directories(dataLoaders_10_1117_windowing
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/datasets/windowed
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(dataLoaders_10_1117_windowing
    PUBLIC
        tensor
        matioCpp
        dataLoaders
        dataLoaders_10_1117
    PRIVATE
        Eigen3::Eigen
)

add_subdirectory(tests)

add_library(data_loaders_fsdd
    STATIC
    loaders/FsddLoader.cpp
    datasets/FsddWindowDataset.cpp
)

target_include_directories(data_loaders_fsdd
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(data_loaders_fsdd
    PUBLIC
        tensor
    PRIVATE
        waveCoreLib
        util
)

nn_enable_target_pch(data_loaders_fsdd)

add_subdirectory(tests)

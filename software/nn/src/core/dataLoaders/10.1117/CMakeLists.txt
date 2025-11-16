add_library(dataLoaders_10_1117
    STATIC
        AudioLoader.cpp
        EEGLoader.cpp
)

target_include_directories(dataLoaders_10_1117
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(dataLoaders_10_1117
    PRIVATE
        Eigen3::Eigen
        ${ZLIB_LIBRARIES}
        matioCpp
)

add_subdirectory(tests)
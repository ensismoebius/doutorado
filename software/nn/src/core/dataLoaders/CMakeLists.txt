find_package(ZLIB REQUIRED)

add_library(dataLoaders
    STATIC
        MatFileUtils.cpp
        DataLoader.cpp
)

target_include_directories(dataLoaders
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(dataLoaders
    PRIVATE
        Eigen3::Eigen
        ${ZLIB_LIBRARIES}
        matioCpp
)

add_subdirectory(tests)

add_subdirectory(10.1117)
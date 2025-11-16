add_library(MatTestUtilsLib STATIC
    MatTestUtils.cpp
)

target_include_directories(MatTestUtilsLib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(MatTestUtilsLib
    PRIVATE
        Eigen3::Eigen
        matioCpp
)
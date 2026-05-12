##
## MatTestUtilsLib
##
## Purpose
## - Small helper library used by loader tests to write MAT v5 fixture files
##   via matio/matio-cpp.
##
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
        tensor
        xtensor xtensor-blas
        matioCpp
)
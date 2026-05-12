add_library(linear_algebra STATIC
    linear_algebra.cpp
)

set_property(TARGET linear_algebra PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(linear_algebra
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_SOURCE_DIR}/src
)

add_subdirectory(tests)

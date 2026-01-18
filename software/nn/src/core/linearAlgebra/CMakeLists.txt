add_library(linearAlgebra STATIC
    linear_algebra.cpp
)

set_property(TARGET linearAlgebra PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(linearAlgebra
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_SOURCE_DIR}/src
)

add_subdirectory(tests)

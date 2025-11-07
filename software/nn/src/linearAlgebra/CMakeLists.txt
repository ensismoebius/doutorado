add_library(linearAlgebra STATIC
    linearAlgebra.cpp
)

target_include_directories(linearAlgebra
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
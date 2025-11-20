include(GoogleTest)

add_library(paraconsistent
    paraconsistent.cpp
)

target_include_directories(paraconsistent
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(paraconsistent
    PUBLIC
        linearAlgebra
        utility
)

add_subdirectory(tests)

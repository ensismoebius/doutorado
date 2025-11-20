include(GoogleTest)

add_library(paraconsistent
    paraconsistent.cpp
)

target_include_directories(paraconsistent
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(paraconsistent
    PRIVATE
        linearAlgebra
        util
)

add_subdirectory(tests)

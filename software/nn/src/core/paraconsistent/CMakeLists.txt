include(GoogleTest)

add_library(paraconsistent
    paraconsistent.cpp
)

target_include_directories(paraconsistent
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(paraconsistent
    PRIVATE
        linear_algebra
        util
)

add_subdirectory(tests)

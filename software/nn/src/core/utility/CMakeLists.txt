
add_library(util
    STATIC
        batching.cpp
        vectorizationCheck.cpp
        synthetic_spike_data.cpp
        comparison.cpp
)

set_property(TARGET util PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(util
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(util
    PRIVATE 
        Eigen3::Eigen
)

add_subdirectory(tests)


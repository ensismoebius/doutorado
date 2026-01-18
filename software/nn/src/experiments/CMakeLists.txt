# Add sanitizer flags
include(../../cmake/SanitizerFlags.cmake)

add_library(experiments_config
    Config.cpp
)

target_sources(experiments_config
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/Config.hpp
)

target_include_directories(experiments_config
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(experiments_config
    PUBLIC
        yaml-cpp
)

# Add subdirectories for all experiments
add_subdirectory(00)
add_subdirectory(01)
add_subdirectory(02)

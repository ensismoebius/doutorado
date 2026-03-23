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

if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json CONFIG REQUIRED)
endif()

target_link_libraries(experiments_config
    PUBLIC
        nlohmann_json::nlohmann_json
)

# Add subdirectories for all experiments
add_subdirectory(00)
add_subdirectory(01)
add_subdirectory(02)
add_subdirectory(03)

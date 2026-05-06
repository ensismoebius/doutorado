##
## VendorYaml.cmake
##
## Purpose
## - Fetch and build `yaml-cpp` (YAML parser/emitter) as a vendored dependency.
##
## What it provides
## - Target: `yaml-cpp` (from upstream).
## - Alias: `YAML::YAML` for consistent namespaced linking from this repo.
##
## Local policy
## - Disable tools/tests/contrib to reduce build surface.
## - Suppress warnings and clang-tidy for vendored code.
##

# VendorYaml.cmake
# Configure vendored yaml-cpp presence and make yaml-cpp available

include(FetchContent)

set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)


FetchContent_MakeAvailable(yaml-cpp)

# Provide an alias for consistency
if(TARGET yaml-cpp)
    add_library(YAML::YAML ALIAS yaml-cpp)
    target_compile_options(yaml-cpp PRIVATE -w)
    set_target_properties(yaml-cpp PROPERTIES CXX_CLANG_TIDY "")
endif()

target_include_directories(yaml-cpp 
    INTERFACE SYSTEM 
        "$<BUILD_INTERFACE:${yaml-cpp_SOURCE_DIR}>"
)
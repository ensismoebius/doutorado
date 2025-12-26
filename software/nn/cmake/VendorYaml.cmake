# VendorYaml.cmake
# Configure vendored yaml-cpp presence and make yaml-cpp available

include(FetchContent)

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        yaml-cpp-0.7.0 # Pinned to a specific tag for reproducibility
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(yaml-cpp)

# Ensure yaml-cpp uses the same C++ standard as the project
set_target_properties(yaml-cpp PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Provide an alias for consistency
if(TARGET yaml-cpp)
    add_library(YAML::YAML ALIAS yaml-cpp)
endif()
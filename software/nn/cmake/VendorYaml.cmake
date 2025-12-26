# VendorYaml.cmake
# Configure vendored yaml-cpp presence and make yaml-cpp available

include(FetchContent)

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        yaml-cpp-0.7.0 
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)


FetchContent_MakeAvailable(yaml-cpp)

# Provide an alias for consistency
if(TARGET yaml-cpp)
    add_library(YAML::YAML ALIAS yaml-cpp)
endif()

target_include_directories(yaml-cpp 
    INTERFACE SYSTEM 
        "$<BUILD_INTERFACE:${yaml-cpp_SOURCE_DIR}>"
)
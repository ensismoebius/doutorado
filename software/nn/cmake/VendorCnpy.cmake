# Cnpy configuration
# Cnpy lets you read and write to .npy and .npz formats in C++.

include(FetchContent)

FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG        4e8810b # Pinned to a specific commit for reproducibility
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
FetchContent_MakeAvailable(cnpy)

target_include_directories(cnpy INTERFACE SYSTEM "$<BUILD_INTERFACE:${cnpy_SOURCE_DIR}>")
target_compile_options(cnpy PRIVATE -w)
target_compile_options(cnpy-static PRIVATE -w)
include(FetchContent)

# Cnpy configuration
# Cnpy lets you read and write to .npy and .npz formats in C++.

FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG        HEAD # Consider using a specific commit hash or tag for reproducibility
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
FetchContent_MakeAvailable(cnpy)
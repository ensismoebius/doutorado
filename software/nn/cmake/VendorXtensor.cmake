# cmake/VendorXtensor.cmake
# Downloads xtl (required by xtensor), xtensor (N-D array), and xtensor-blas (matmul).
# All are header-only except xtensor-blas which requires BLAS (already found via PackageChecking).

include(FetchContent)

# xtl — type utilities required by xtensor
FetchContent_Declare(xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    GIT_TAG        0.7.7
    GIT_SHALLOW    TRUE)

# xtensor — core N-D array library
FetchContent_Declare(xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    GIT_TAG        0.25.0
    GIT_SHALLOW    TRUE)

# xtensor-blas — BLAS-backed linalg (dot, norm, solve)
FetchContent_Declare(xtensor-blas
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas
    GIT_TAG        0.21.0
    GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(xtl xtensor xtensor-blas)

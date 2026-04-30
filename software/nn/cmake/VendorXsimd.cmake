# cmake/VendorXsimd.cmake
# Integrates xsimd for SIMD acceleration in xtensor.
# xsimd is a header-only library.

include(FetchContent)

FetchContent_Declare(xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd
    GIT_TAG        14.2.0
    GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(xsimd)

# Link the xsimd target to the project's tensor backend.
# Since xsimd is header-only, this mostly ensures include paths are set.
if(TARGET xsimd)
    # We'll use this target in the tensor CMakeLists.txt
endif()

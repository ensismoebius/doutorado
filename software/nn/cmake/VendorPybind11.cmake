##
## VendorPybind11.cmake
##
## Provides pybind11 for the optional `nn_microscope` Python extension
## (src/bindings/), consumed by software/experiment_microscope/.
##
## Only active when NN_BUILD_PY_BINDINGS=ON (set by the `python-bindings`
## preset). A no-op otherwise, so ordinary builds pay nothing.
##
## Vendored via FetchContent with a pinned tag (mirrors VendorMatplotlibCpp /
## VendorIncludes). FIND_PACKAGE_ARGS lets an already-installed pybind11 of a
## compatible version satisfy it without a clone; the pin is the reproducible
## floor.
##

if(NOT NN_BUILD_PY_BINDINGS)
    return()
endif()

include(FetchContent)

# Python interpreter + development headers + NumPy are already hard build deps
# via VendorMatplotlibCpp.cmake; request them explicitly here so this file is
# self-contained if that ever changes.
find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

set(NN_PYBIND11_TAG "v3.1.0" CACHE STRING "Pinned pybind11 git tag for nn_microscope")

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        ${NN_PYBIND11_TAG}
    GIT_SHALLOW    TRUE
    FIND_PACKAGE_ARGS 2.13
)

FetchContent_MakeAvailable(pybind11)

if(NOT TARGET pybind11::module)
    message(FATAL_ERROR "VendorPybind11: pybind11 target not available after FetchContent.")
endif()

message(STATUS "nn_microscope: pybind11 ready (tag floor ${NN_PYBIND11_TAG})")

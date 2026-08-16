##
## VendorMatplotlibCpp.cmake
##
## Purpose
## - Fetch `matplotlib-cpp` (header-only plotting wrapper) and make it usable from CMake.
## - Optionally bootstrap a Python venv containing NumPy + Matplotlib for demos/tests.
##
## What it provides
## - Target: `MatplotlibCpp` (INTERFACE) and alias `MatplotlibCpp::MatplotlibCpp`.
## - Includes: matplotlib-cpp headers + Python + NumPy include dirs.
##
## Notes / pitfalls
## - When `MATPLOTLIBCPP_CREATE_VENV=ON`, this runs `python -m venv` and `pip install` at
##   *configure* time and sets `Python3_EXECUTABLE` in the cache to point to the venv.
##   This is convenient locally but can be undesirable in hermetic CI.
## - `matplotlib_cpp` (lowercase) may also be defined by the upstream FetchContent project;
##   this file disables clang-tidy for that target as it is third-party code.
##

# ------------------------------------------------------------
# VendorMatplotlibCpp.cmake
# Header-only matplotlib-cpp + Python/NumPy provisioning
# ------------------------------------------------------------

include(FetchContent)

option(
    MATPLOTLIBCPP_CREATE_VENV
    "Create a Python venv and install matplotlib/numpy during configure"
    ON
)

set(MATPLOTLIBCPP_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/matplotlib-cpp-src")
set(MATPLOTLIBCPP_VENV_DIR "${CMAKE_BINARY_DIR}/venv")

# ------------------------------------------------------------
# Fetch matplotlib-cpp (header-only)
# ------------------------------------------------------------
# Disable examples to avoid warnings
set(MATPLOTLIB_CPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Use a minimal CMakeLists.txt to replace the upstream one
# This avoids building examples and fixes include paths
set(MATPLOTLIBCPP_PATCH_SCRIPT
    "${CMAKE_SOURCE_DIR}/cmake/patches/matplotlib_cpp/apply_patches.cmake")

FetchContent_Declare(
    matplotlib_cpp
    GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
    GIT_TAG        ef0383f1315d32e0156335e10b82e90b334f6d9f
    SOURCE_DIR     "${MATPLOTLIBCPP_SRC_DIR}"
    PATCH_COMMAND  "${CMAKE_COMMAND}"
        -DMATPLOTLIBCPP_SOURCE_DIR=${MATPLOTLIBCPP_SRC_DIR}
        -P "${MATPLOTLIBCPP_PATCH_SCRIPT}"
)

FetchContent_MakeAvailable(matplotlib_cpp)

if(NOT EXISTS "${MATPLOTLIBCPP_SRC_DIR}/matplotlibcpp.h")
    message(FATAL_ERROR "matplotlib-cpp headers not found after FetchContent.")
endif()

# ------------------------------------------------------------
# Find system Python (bootstrap)
# ------------------------------------------------------------
# NOTE: this whole bootstrap block MUST run before any find_package(Python3 ...)
# call below. find_package() caches Python3_EXECUTABLE; if a prior configure
# left a broken/partial venv (interrupted run, deleted venv dir, system Python
# upgrade invalidating the venv's interpreter symlink), a cached-but-broken
# path makes find_package() fail fatally before we ever get a chance to
# detect and repair it.
find_program(_SYSTEM_PYTHON_EXECUTABLE NAMES python3 python REQUIRED)

# ------------------------------------------------------------
# Resolve venv python path (portable)
# ------------------------------------------------------------
if(WIN32)
    set(_VENV_PY "${MATPLOTLIBCPP_VENV_DIR}/Scripts/python.exe")
else()
    set(_VENV_PY "${MATPLOTLIBCPP_VENV_DIR}/bin/python")
endif()

# ------------------------------------------------------------
# Create (or repair) venv if requested
# ------------------------------------------------------------
if(MATPLOTLIBCPP_CREATE_VENV)
    set(_venv_needs_bootstrap FALSE)

    if(NOT EXISTS "${_VENV_PY}")
        set(_venv_needs_bootstrap TRUE)
    else()
        # EXISTS alone isn't enough: a broken symlink (e.g. venv created
        # against a system Python that was since upgraded/removed) or a
        # partial venv from an interrupted configure can leave a path that
        # exists but doesn't run. Probe it and rebuild if it's dead.
        execute_process(
            COMMAND "${_VENV_PY}" -c "import sys"
            RESULT_VARIABLE _venv_probe_res
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(NOT _venv_probe_res EQUAL 0)
            message(WARNING "MATPLOTLIBCPP: existing venv at ${MATPLOTLIBCPP_VENV_DIR} is broken, recreating")
            file(REMOVE_RECURSE "${MATPLOTLIBCPP_VENV_DIR}")
            set(_venv_needs_bootstrap TRUE)
        endif()
    endif()

    if(_venv_needs_bootstrap)
        message(STATUS "MATPLOTLIBCPP: creating venv at ${MATPLOTLIBCPP_VENV_DIR}")

        # `python -m venv` does not overwrite files that already exist, so a
        # stale/partial tree left behind by an earlier interrupted or broken
        # bootstrap (e.g. a half-copied `pip` package missing __main__.py)
        # would otherwise persist across repeated "repair" attempts.
        file(REMOVE_RECURSE "${MATPLOTLIBCPP_VENV_DIR}")

        execute_process(
            COMMAND "${_SYSTEM_PYTHON_EXECUTABLE}" -m venv --clear "${MATPLOTLIBCPP_VENV_DIR}"
            RESULT_VARIABLE _venv_res
        )
        if(NOT _venv_res EQUAL 0)
            message(FATAL_ERROR "Failed to create Python venv.")
        endif()

        execute_process(
            COMMAND "${_VENV_PY}" -m pip install --upgrade pip setuptools wheel
            RESULT_VARIABLE _pip_upd_res
        )
        if(NOT _pip_upd_res EQUAL 0)
            message(FATAL_ERROR "Failed to upgrade pip in venv.")
        endif()

        execute_process(
            COMMAND "${_VENV_PY}" -m pip install numpy matplotlib
            RESULT_VARIABLE _pip_inst_res
        )
        if(NOT _pip_inst_res EQUAL 0)
            message(FATAL_ERROR "Failed to install numpy/matplotlib in venv.")
        endif()
    endif()

    set(Python3_EXECUTABLE "${_VENV_PY}" CACHE FILEPATH "Python from matplotlib-cpp venv" FORCE)
else()
    set(Python3_EXECUTABLE "${_SYSTEM_PYTHON_EXECUTABLE}" CACHE FILEPATH "System Python" FORCE)
endif()

# ------------------------------------------------------------
# Find Python + NumPy (hard requirement)
# ------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

if(NOT Python3_NumPy_INCLUDE_DIRS)
    message(FATAL_ERROR "NumPy headers not found.")
endif()

# Configure the target (defined by FetchContent_MakeAvailable)
if(TARGET matplotlib_cpp)
    target_compile_features(matplotlib_cpp INTERFACE cxx_std_11)

    # CMP0079 allows linking to targets created in other directories
    cmake_policy(PUSH)
    if(POLICY CMP0079)
        cmake_policy(SET CMP0079 NEW)
    endif()
    target_link_libraries(matplotlib_cpp INTERFACE Python3::Python Python3::Module)

    if(Python3_NumPy_FOUND)
        target_link_libraries(matplotlib_cpp INTERFACE Python3::NumPy)
    else()
        target_compile_definitions(matplotlib_cpp INTERFACE WITHOUT_NUMPY)
    endif()
    cmake_policy(POP)

    # Suppress clang-tidy
    set_target_properties(matplotlib_cpp PROPERTIES CXX_CLANG_TIDY "")
endif()

# ------------------------------------------------------------
# Create INTERFACE target (REAL target)
# ------------------------------------------------------------
add_library(MatplotlibCpp INTERFACE)

target_include_directories(MatplotlibCpp
    SYSTEM INTERFACE
        "${MATPLOTLIBCPP_SRC_DIR}"
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
)

if(TARGET Python3::Python)
    target_link_libraries(MatplotlibCpp INTERFACE Python3::Python)
elseif(Python3_LIBRARIES)
    target_link_libraries(MatplotlibCpp INTERFACE ${Python3_LIBRARIES})
endif()

if(TARGET Python3::NumPy)
    target_link_libraries(MatplotlibCpp INTERFACE Python3::NumPy)
endif()

# ------------------------------------------------------------
# Public ALIAS (canonical, namespaced)
# ------------------------------------------------------------
add_library(MatplotlibCpp::MatplotlibCpp ALIAS MatplotlibCpp)
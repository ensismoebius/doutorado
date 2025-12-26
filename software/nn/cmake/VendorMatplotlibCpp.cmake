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
FetchContent_Declare(
    matplotlib_cpp
    GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
    GIT_TAG        ef0383f1315d32e0156335e10b82e90b334f6d9f
    SOURCE_DIR     "${MATPLOTLIBCPP_SRC_DIR}"
)
FetchContent_MakeAvailable(matplotlib_cpp)

if(NOT EXISTS "${MATPLOTLIBCPP_SRC_DIR}/matplotlibcpp.h")
    message(FATAL_ERROR "matplotlib-cpp headers not found after FetchContent.")
endif()

# ------------------------------------------------------------
# Find system Python (bootstrap)
# ------------------------------------------------------------
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
# Create venv if requested
# ------------------------------------------------------------
if(MATPLOTLIBCPP_CREATE_VENV)
    if(NOT EXISTS "${_VENV_PY}")
        message(STATUS "MATPLOTLIBCPP: creating venv at ${MATPLOTLIBCPP_VENV_DIR}")

        execute_process(
            COMMAND "${_SYSTEM_PYTHON_EXECUTABLE}" -m venv "${MATPLOTLIBCPP_VENV_DIR}"
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
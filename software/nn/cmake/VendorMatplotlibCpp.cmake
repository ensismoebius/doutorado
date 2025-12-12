# VendorMatplotlibCpp.cmake
# Configure matplotlib-cpp (header-only), criar venv opcionalmente e garantir Python/NumPy no venv.

include(FetchContent)

# opção para criar venv durante o configure
option(MATPLOTLIBCPP_CREATE_VENV "Create venv and install matplotlib/numpy during CMake configure" ON)
set(MATPLOTLIBCPP_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/matplotlib-cpp-install")
set(MATPLOTLIBCPP_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/matplotlib-cpp-src")
set(MATPLOTLIBCPP_VENV_DIR "${CMAKE_BINARY_DIR}/venv")
set(MATPLOTLIBCPP_VENV_PY "${MATPLOTLIBCPP_VENV_DIR}/bin/python")

# --- Fetch matplotlib-cpp (header-only) ---
FetchContent_Declare(
  matplotlib_cpp
  GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
  GIT_TAG        ef0383f1315d32e0156335e10b82e90b334f6d9f
  SOURCE_DIR     "${MATPLOTLIBCPP_SRC_DIR}"
)
FetchContent_MakeAvailable(matplotlib_cpp)

# --- Find a system Python executable to create venv if needed ---
find_program(_SYSTEM_PYTHON_EXECUTABLE NAMES python3 python REQUIRED)
if(NOT _SYSTEM_PYTHON_EXECUTABLE)
    message(FATAL_ERROR "No system Python found (tried python3, python). Required to create venv.")
endif()

# --- Create venv and install packages if requested ---
if(MATPLOTLIBCPP_CREATE_VENV)
    # se existir mas não executável, remove e recria
    if(EXISTS "${MATPLOTLIBCPP_VENV_PY}" AND NOT IS_EXECUTABLE "${MATPLOTLIBCPP_VENV_PY}")
        file(REMOVE_RECURSE "${MATPLOTLIBCPP_VENV_DIR}")
        message(STATUS "MATPLOTLIBCPP: venv inválido removido, será recriado.")
    endif()

    if(NOT EXISTS "${MATPLOTLIBCPP_VENV_PY}")
        message(STATUS "MATPLOTLIBCPP: criando venv em ${MATPLOTLIBCPP_VENV_DIR} usando ${_SYSTEM_PYTHON_EXECUTABLE}")
        execute_process(
            COMMAND "${_SYSTEM_PYTHON_EXECUTABLE}" -m venv "${MATPLOTLIBCPP_VENV_DIR}"
            RESULT_VARIABLE _venv_res
            OUTPUT_VARIABLE _venv_out
            ERROR_VARIABLE _venv_err
            TIMEOUT 60
        )
        if(NOT _venv_res EQUAL 0)
            message(FATAL_ERROR "MATPLOTLIBCPP: falha criando venv: ${_venv_err}\nOUTPUT:\n${_venv_out}")
        endif()

        execute_process(
            COMMAND "${MATPLOTLIBCPP_VENV_PY}" -m pip install --upgrade pip setuptools wheel
            RESULT_VARIABLE _pip_upd_res
            OUTPUT_VARIABLE _pip_upd_out
            ERROR_VARIABLE _pip_upd_err
            TIMEOUT 120
        )
        if(NOT _pip_upd_res EQUAL 0)
            message(FATAL_ERROR "MATPLOTLIBCPP: falha upgrade pip: ${_pip_upd_err}\n${_pip_upd_out}")
        endif()

        execute_process(
            COMMAND "${MATPLOTLIBCPP_VENV_PY}" -m pip install numpy matplotlib
            RESULT_VARIABLE _pip_inst_res
            OUTPUT_VARIABLE _pip_inst_out
            ERROR_VARIABLE _pip_inst_err
            TIMEOUT 300
        )
        if(NOT _pip_inst_res EQUAL 0)
            message(FATAL_ERROR "MATPLOTLIBCPP: falha pip install numpy/matplotlib: ${_pip_inst_err}\n${_pip_inst_out}")
        endif()
    else()
        message(STATUS "MATPLOTLIBCPP: venv existente e executável em ${MATPLOTLIBCPP_VENV_PY}")
    endif()

    if(EXISTS "${MATPLOTLIBCPP_VENV_PY}" AND IS_EXECUTABLE "${MATPLOTLIBCPP_VENV_PY}")
        set(Python3_EXECUTABLE "${MATPLOTLIBCPP_VENV_PY}" CACHE PATH "Python3 executable (venv)" FORCE)
        message(STATUS "MATPLOTLIBCPP: will use venv python ${Python3_EXECUTABLE} for find_package(Python3 ...)")
    else()
        message(FATAL_ERROR "MATPLOTLIBCPP: após tentativas, venv python não está disponível/executável.")
    endif()
else()
    # user opted out: use system python
    set(Python3_EXECUTABLE "${_SYSTEM_PYTHON_EXECUTABLE}" CACHE PATH "Python3 executable (system)" FORCE)
    message(STATUS "MATPLOTLIBCPP_CREATE_VENV=OFF: using system python ${Python3_EXECUTABLE}")
endif()

# --- Now find Python and NumPy using FindPython ---
# Request Interpreter, Development and NumPy components
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# --- Create imported interface library for matplotlib-cpp (GLOBAL) ---
# Mark as GLOBAL to ensure visibility across subdirectories
add_library(MatplotlibCpp::MatplotlibCpp INTERFACE IMPORTED GLOBAL)

# --- Prepare include paths: matplotlib-cpp root + Python + NumPy includes ---
set(_mplcpp_include "${MATPLOTLIBCPP_SRC_DIR}")
set(_python_inc "${Python3_INCLUDE_DIRS}")
set(_numpy_inc "${Python3_NumPy_INCLUDE_DIRS}")

if(NOT EXISTS "${_mplcpp_include}")
    message(FATAL_ERROR "matplotlib-cpp source not found at ${_mplcpp_include}. FetchContent should have populated it.")
endif()

set_target_properties(MatplotlibCpp::MatplotlibCpp PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_mplcpp_include};${_python_inc};${_numpy_inc}"
)

# Prefer linking to imported Python target if available
if(TARGET Python3::Python)
    target_link_libraries(MatplotlibCpp::MatplotlibCpp INTERFACE Python3::Python)
elseif(Python3_LIBRARIES)
    set_target_properties(MatplotlibCpp::MatplotlibCpp PROPERTIES
        INTERFACE_LINK_LIBRARIES "${Python3_LIBRARIES}"
    )
endif()

# If NumPy target exists, link it
if(TARGET Python3::NumPy)
    target_link_libraries(MatplotlibCpp::MatplotlibCpp INTERFACE Python3::NumPy)
endif()

message(STATUS "Matplotlib-cpp configured. Include dir: ${_mplcpp_include}")

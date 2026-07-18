##
## VendorPythonEnv.cmake
##
## Purpose
## - Bootstrap the project-level Python venv (`.venv` at the repo root) used by
##   pipeline/testing scripts (numpy, scipy, torch, snntorch, ...) so
##   `scripts/pipeline/**/*.sh` and `scripts/testing/gen_*_refs.py` don't depend on
##   whatever bare `python3` happens to resolve to on the system PATH.
##
## Notes / pitfalls
## - This is a SEPARATE venv from the one in VendorMatplotlibCpp.cmake: that one lives
##   under CMAKE_BINARY_DIR and is linked into C++ demos via Python3::Python/NumPy. This
##   one lives at CMAKE_SOURCE_DIR/.venv (gitignored) and is invoked as an external
##   interpreter by shell/python pipeline scripts — see scripts/requirements.txt.
## - Reinstalling requirements on every configure would make `cmake --preset=...`
##   painfully slow (torch is a multi-hundred-MB download even CPU-only). A stamp file
##   records the SHA-256 of requirements.txt; pip install only reruns when that hash
##   changes or the venv is missing/broken.
##

option(
    PYTHON_ENV_CREATE_VENV
    "Create/maintain the project .venv (numpy/scipy/torch/...) during configure"
    ON
)

set(PYTHON_ENV_DIR "${CMAKE_SOURCE_DIR}/.venv" CACHE PATH "Project Python venv directory")
set(PYTHON_ENV_REQUIREMENTS "${CMAKE_SOURCE_DIR}/scripts/requirements.txt" CACHE FILEPATH
    "Requirements file installed into the project venv")

if(PYTHON_ENV_CREATE_VENV)
    find_program(_PYENV_SYSTEM_PYTHON NAMES python3 python REQUIRED)

    if(WIN32)
        set(_PYENV_PY "${PYTHON_ENV_DIR}/Scripts/python.exe")
    else()
        set(_PYENV_PY "${PYTHON_ENV_DIR}/bin/python3")
    endif()

    # EXISTS alone isn't enough: a broken symlink (venv created against a system Python
    # that was since upgraded/removed) or a partial venv from an interrupted configure
    # can leave a path that exists but doesn't run. Probe it and rebuild if it's dead.
    set(_pyenv_needs_bootstrap FALSE)
    if(NOT EXISTS "${_PYENV_PY}")
        set(_pyenv_needs_bootstrap TRUE)
    else()
        execute_process(
            COMMAND "${_PYENV_PY}" -c "import sys"
            RESULT_VARIABLE _pyenv_probe_res
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(NOT _pyenv_probe_res EQUAL 0)
            message(WARNING "PYTHON_ENV: existing venv at ${PYTHON_ENV_DIR} is broken, recreating")
            file(REMOVE_RECURSE "${PYTHON_ENV_DIR}")
            set(_pyenv_needs_bootstrap TRUE)
        endif()
    endif()

    if(NOT EXISTS "${PYTHON_ENV_REQUIREMENTS}")
        message(FATAL_ERROR "PYTHON_ENV: requirements file not found: ${PYTHON_ENV_REQUIREMENTS}")
    endif()

    file(SHA256 "${PYTHON_ENV_REQUIREMENTS}" _pyenv_req_hash)
    set(_pyenv_stamp "${PYTHON_ENV_DIR}/.requirements.sha256")

    set(_pyenv_needs_install ${_pyenv_needs_bootstrap})
    if(NOT _pyenv_needs_install)
        if(NOT EXISTS "${_pyenv_stamp}")
            set(_pyenv_needs_install TRUE)
        else()
            file(READ "${_pyenv_stamp}" _pyenv_stamp_hash)
            string(STRIP "${_pyenv_stamp_hash}" _pyenv_stamp_hash)
            if(NOT _pyenv_stamp_hash STREQUAL _pyenv_req_hash)
                set(_pyenv_needs_install TRUE)
            endif()
        endif()
    endif()

    if(_pyenv_needs_bootstrap)
        message(STATUS "PYTHON_ENV: creating venv at ${PYTHON_ENV_DIR}")

        # `python -m venv` does not overwrite files that already exist, so a
        # stale/partial tree left behind by an earlier interrupted or broken
        # bootstrap would otherwise persist across repeated "repair" attempts.
        file(REMOVE_RECURSE "${PYTHON_ENV_DIR}")

        execute_process(
            COMMAND "${_PYENV_SYSTEM_PYTHON}" -m venv --clear "${PYTHON_ENV_DIR}"
            RESULT_VARIABLE _pyenv_venv_res
        )
        if(NOT _pyenv_venv_res EQUAL 0)
            message(FATAL_ERROR "PYTHON_ENV: failed to create venv at ${PYTHON_ENV_DIR}.")
        endif()

        execute_process(
            COMMAND "${_PYENV_PY}" -m pip install --upgrade pip setuptools wheel
            RESULT_VARIABLE _pyenv_pip_upd_res
        )
        if(NOT _pyenv_pip_upd_res EQUAL 0)
            message(FATAL_ERROR "PYTHON_ENV: failed to upgrade pip in venv.")
        endif()
    endif()

    if(_pyenv_needs_install)
        message(STATUS "PYTHON_ENV: installing ${PYTHON_ENV_REQUIREMENTS} into ${PYTHON_ENV_DIR} (this can take a while the first time — CPU-only torch is ~200MB)")
        execute_process(
            COMMAND "${_PYENV_PY}" -m pip install -r "${PYTHON_ENV_REQUIREMENTS}"
            RESULT_VARIABLE _pyenv_pip_inst_res
        )
        if(NOT _pyenv_pip_inst_res EQUAL 0)
            message(FATAL_ERROR "PYTHON_ENV: failed to install ${PYTHON_ENV_REQUIREMENTS}.")
        endif()
        file(WRITE "${_pyenv_stamp}" "${_pyenv_req_hash}")
    else()
        message(STATUS "PYTHON_ENV: venv up to date at ${PYTHON_ENV_DIR}")
    endif()

    set(PYTHON_ENV_EXECUTABLE "${_PYENV_PY}" CACHE FILEPATH "Python executable inside the project venv" FORCE)
endif()

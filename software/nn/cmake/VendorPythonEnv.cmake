##
## VendorPythonEnv.cmake
##
## Purpose
## - Provide the Python venv (default: the machine's single shared `$HOME/.venv`) used by
##   pipeline/testing scripts (numpy, scipy, torch, snntorch, ...) so
##   `scripts/pipeline/**/*.sh` and `scripts/testing/gen_*_refs.py` don't depend on
##   whatever bare `python3` happens to resolve to on the system PATH.
##
## Notes / pitfalls
## - By default this is the SAME venv as VendorMatplotlibCpp.cmake's: `$HOME/.venv`, the
##   machine's one shared Python env (disk space: one torch install, not one per checkout).
##   It is invoked as an external interpreter by shell/python pipeline scripts — see
##   scripts/requirements.txt. Override with -DPYTHON_ENV_DIR=... ; a venv outside this
##   source/build tree is only created when missing, never deleted or `--clear`ed.
## - Reinstalling requirements on every configure would make `cmake --preset=...`
##   painfully slow (torch is a multi-hundred-MB download even CPU-only). A stamp file
##   records the SHA-256 of requirements.txt; pip install only reruns when that hash
##   changes or the venv is missing/broken.
##

option(
    PYTHON_ENV_CREATE_VENV
    "Create/maintain the Python venv (default shared ~/.venv; numpy/scipy/torch/...) during configure"
    ON
)

if(NOT "$ENV{HOME}" STREQUAL "")
    set(_pyenv_default_dir "$ENV{HOME}/.venv")
else()
    set(_pyenv_default_dir "${CMAKE_SOURCE_DIR}/.venv")
endif()
set(PYTHON_ENV_DIR "${_pyenv_default_dir}" CACHE PATH "Python venv directory (default: shared ~/.venv)")
# Build dirs configured before the switch to the shared env cached the old in-repo default.
if(PYTHON_ENV_DIR STREQUAL "${CMAKE_SOURCE_DIR}/.venv" AND NOT EXISTS "${CMAKE_SOURCE_DIR}/.venv")
    set(PYTHON_ENV_DIR "${_pyenv_default_dir}" CACHE PATH "Python venv directory (default: shared ~/.venv)" FORCE)
endif()
# Only a venv inside this source/build tree may be deleted and recreated. A shared
# one outside it (default: ~/.venv, the machine's single Python env used by every
# tool) is never removed or `--clear`ed — a broken one is a FATAL_ERROR instead.
string(FIND "${PYTHON_ENV_DIR}/" "${CMAKE_SOURCE_DIR}/" _pyenv_in_src)
string(FIND "${PYTHON_ENV_DIR}/" "${CMAKE_BINARY_DIR}/" _pyenv_in_bin)
if(_pyenv_in_src EQUAL 0 OR _pyenv_in_bin EQUAL 0)
    set(_pyenv_owned TRUE)
else()
    set(_pyenv_owned FALSE)
endif()
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
            if(NOT _pyenv_owned)
                message(FATAL_ERROR "PYTHON_ENV: shared venv at ${PYTHON_ENV_DIR} is broken. "
                    "Refusing to delete it (other tools use it); repair it by hand, e.g. "
                    "`python3 -m venv --upgrade ${PYTHON_ENV_DIR}`.")
            endif()
            message(WARNING "PYTHON_ENV: existing venv at ${PYTHON_ENV_DIR} is broken, recreating")
            file(REMOVE_RECURSE "${PYTHON_ENV_DIR}")
            set(_pyenv_needs_bootstrap TRUE)
        endif()
    endif()

    if(NOT EXISTS "${PYTHON_ENV_REQUIREMENTS}")
        message(FATAL_ERROR "PYTHON_ENV: requirements file not found: ${PYTHON_ENV_REQUIREMENTS}")
    endif()

    file(SHA256 "${PYTHON_ENV_REQUIREMENTS}" _pyenv_req_hash)
    # Project-named: the shared venv may carry stamps from several projects.
    set(_pyenv_stamp "${PYTHON_ENV_DIR}/.${PROJECT_NAME}-requirements.sha256")

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
        if(_pyenv_owned)
            file(REMOVE_RECURSE "${PYTHON_ENV_DIR}")
            set(_pyenv_clear --clear)
        else()
            set(_pyenv_clear "")
        endif()

        execute_process(
            COMMAND "${_PYENV_SYSTEM_PYTHON}" -m venv ${_pyenv_clear} "${PYTHON_ENV_DIR}"
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

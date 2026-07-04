##
# VendorSqlite.cmake
#
# Provides a small optional integration for a vendored SQLite amalgamation.
# If the directory `third_party/sqlite` contains `sqlite3.c` and `sqlite3.h`
# this module will create a static target `sqlite3_vendored` that other
# targets can link to. The build uses the current project's compiler flags
# and will add ASan-related compile/link options when `NN_ENABLE_ASAN` is set.
##

option(USE_VENDORED_SQLITE "Force building and using vendored SQLite" OFF)

# Prefer system SQLite when available; fall back to vendored amalgamation.
set(VENDORED_SQLITE_DIR ${CMAKE_SOURCE_DIR}/third_party/sqlite)
set(SQLITE_SRC ${VENDORED_SQLITE_DIR}/sqlite3.c)
set(SQLITE_HDR ${VENDORED_SQLITE_DIR}/sqlite3.h)

# Try system package first unless explicitly forced to use vendored.
if(NOT USE_VENDORED_SQLITE)
    find_package(SQLite3 QUIET)
    if(SQLite3_FOUND)
        # Normalize to canonical target name expected by project code.
        # Some environments expose SQLite::SQLite3 or only variables.
        if(NOT TARGET SQLite3::SQLite3)
            if(TARGET SQLite::SQLite3)
                add_library(SQLite3::SQLite3 INTERFACE IMPORTED)
                target_link_libraries(SQLite3::SQLite3 INTERFACE SQLite::SQLite3)
            elseif(SQLite3_LIBRARIES OR SQLite3_LIBRARY)
                add_library(SQLite3::SQLite3 INTERFACE IMPORTED)
                if(SQLite3_LIBRARIES)
                    target_link_libraries(SQLite3::SQLite3 INTERFACE ${SQLite3_LIBRARIES})
                else()
                    target_link_libraries(SQLite3::SQLite3 INTERFACE ${SQLite3_LIBRARY})
                endif()

                if(SQLite3_INCLUDE_DIRS)
                    set_property(TARGET SQLite3::SQLite3 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${SQLite3_INCLUDE_DIRS}")
                elseif(SQLite3_INCLUDE_DIR)
                    set_property(TARGET SQLite3::SQLite3 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${SQLite3_INCLUDE_DIR}")
                endif()
            endif()
        endif()

        if(TARGET SQLite3::SQLite3)
            message(STATUS "Found system SQLite3 (preferred). Using system package")
            return()
        endif()

        message(STATUS "System SQLite3 found but no usable CMake target; will attempt vendored SQLite")
    else()
        message(STATUS "System SQLite3 not found; will attempt vendored SQLite if available")
    endif()
else()
    message(STATUS "USE_VENDORED_SQLITE=ON: forcing vendored SQLite build")
endif()

# Optional: allow FetchContent download if amalgamation isn't checked in.
# Set `SQLITE_FETCH_URL` to override the upstream archive URL if desired.
if(NOT DEFINED SQLITE_FETCH_URL)
    # Default amalgamation version (can be overridden from the top-level
    # configure command line). This encodes the SQLite release: 3.51.3 -> 3510300
    set(SQLITE_AMALGAMATION_VERSION "3510300" CACHE STRING "SQLite amalgamation version code")
    set(SQLITE_FETCH_URL "https://www.sqlite.org/sqlite-amalgamation-${SQLITE_AMALGAMATION_VERSION}.zip")
endif()

# If the amalgamation is missing, try to download a known upstream archive
# from a few sqlite.org hostnames, extract it to the build tree, and use
# the extracted sqlite3.c / sqlite3.h. If that fails, fall back to
# FetchContent as a last resort.
if(NOT (EXISTS ${SQLITE_SRC} AND EXISTS ${SQLITE_HDR}))
    set(_sqlite_amalg_ok FALSE)
    set(_sqlite_zip "${CMAKE_BINARY_DIR}/sqlite-amalgamation-${SQLITE_AMALGAMATION_VERSION}.zip")
    set(_sqlite_extract_dir "${CMAKE_BINARY_DIR}/_deps/sqlite_amalgamation")

    # Prepare a list of full URLs to try. Use sqlite.org mirrors first,
    # then GitHub archive URLs for the corresponding tag.
    if(NOT DEFINED SQLITE_GH_TAG)
        set(SQLITE_GH_TAG "version-3.51.3" CACHE STRING "GitHub tag to try for sqlite mirror archives")
    endif()
    set(_sqlite_urls
        "https://www.sqlite.org/sqlite-amalgamation-${SQLITE_AMALGAMATION_VERSION}.zip"
        "https://sqlite.org/sqlite-amalgamation-${SQLITE_AMALGAMATION_VERSION}.zip"
        "https://www2.sqlite.org/sqlite-amalgamation-${SQLITE_AMALGAMATION_VERSION}.zip"
        "https://www3.sqlite.org/sqlite-amalgamation-${SQLITE_AMALGAMATION_VERSION}.zip"
        "https://github.com/sqlite/sqlite/archive/refs/tags/${SQLITE_GH_TAG}.zip"
        "https://github.com/sqlite/sqlite/archive/refs/tags/${SQLITE_GH_TAG}.tar.gz"
    )

    foreach(_try_url IN LISTS _sqlite_urls)
        message(STATUS "Attempting download of SQLite amalgamation from ${_try_url}")
        file(DOWNLOAD "${_try_url}" "${_sqlite_zip}" STATUS _dl_status SHOW_PROGRESS)
        list(GET _dl_status 0 _dl_code)
        if(_dl_code EQUAL 0)
            message(STATUS "Downloaded amalgamation from ${_try_url} to ${_sqlite_zip}")
            file(MAKE_DIRECTORY "${_sqlite_extract_dir}")
            file(ARCHIVE_EXTRACT INPUT "${_sqlite_zip}" DESTINATION "${_sqlite_extract_dir}")
            # Try to locate the extracted sqlite3.c
            file(GLOB_RECURSE _found_c "${_sqlite_extract_dir}/*sqlite3.c")
            list(LENGTH _found_c _found_len)
            if(_found_len GREATER 0)
                list(GET _found_c 0 _first_c)
                get_filename_component(SQLITE_SRC "${_first_c}" ABSOLUTE)
                get_filename_component(_hdr_dir "${_first_c}" DIRECTORY)
                set(SQLITE_HDR "${_hdr_dir}/sqlite3.h")
                set(VENDORED_SQLITE_DIR "${_hdr_dir}")
                set(_sqlite_amalg_ok TRUE)
                break()
            else()
                message(STATUS "Downloaded archive but did not find sqlite3.c inside extracted content; attempting to build amalgamation from source")

                # Try to locate a directory with a top-level Makefile to run `make sqlite3.c`.
                file(GLOB_RECURSE _make_files RELATIVE "${_sqlite_extract_dir}" "${_sqlite_extract_dir}/Makefile" "${_sqlite_extract_dir}/*/Makefile")
                list(LENGTH _make_files _make_len)
                if(_make_len GREATER 0)
                    list(GET _make_files 0 _make_rel)
                    get_filename_component(_make_dir "${_sqlite_extract_dir}/${_make_rel}" DIRECTORY)
                else()
                    set(_make_dir "${_sqlite_extract_dir}")
                endif()

                find_program(_MAKE_PROG make)
                if(_MAKE_PROG)
                    message(STATUS "Found make at ${_MAKE_PROG}; running make in ${_make_dir}")
                    execute_process(COMMAND ${_MAKE_PROG} sqlite3.c
                                    WORKING_DIRECTORY "${_make_dir}"
                                    RESULT_VARIABLE _mk_res
                                    OUTPUT_VARIABLE _mk_out
                                    ERROR_VARIABLE _mk_err
                                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
                    if(NOT _mk_res EQUAL 0)
                        message(STATUS "make sqlite3.c failed; trying 'make target_source && make sqlite3.c'")
                        execute_process(COMMAND ${_MAKE_PROG} target_source
                                        WORKING_DIRECTORY "${_make_dir}"
                                        RESULT_VARIABLE _mk_res2
                                        OUTPUT_VARIABLE _mk_out2
                                        ERROR_VARIABLE _mk_err2
                                        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
                        if(_mk_res2 EQUAL 0)
                            execute_process(COMMAND ${_MAKE_PROG} sqlite3.c
                                            WORKING_DIRECTORY "${_make_dir}"
                                            RESULT_VARIABLE _mk_res3
                                            OUTPUT_VARIABLE _mk_out3
                                            ERROR_VARIABLE _mk_err3
                                            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
                        endif()
                    endif()

                    # Re-scan for sqlite3.c after attempted build
                    file(GLOB_RECURSE _found_c_after "${_sqlite_extract_dir}/*sqlite3.c")
                    list(LENGTH _found_c_after _found_after_len)
                    if(_found_after_len GREATER 0)
                        list(GET _found_c_after 0 _first_c)
                        get_filename_component(SQLITE_SRC "${_first_c}" ABSOLUTE)
                        get_filename_component(_hdr_dir "${_first_c}" DIRECTORY)
                        set(SQLITE_HDR "${_hdr_dir}/sqlite3.h")
                        set(VENDORED_SQLITE_DIR "${_hdr_dir}")
                        set(_sqlite_amalg_ok TRUE)
                        break()
                    else()
                        message(WARNING "Could not build sqlite3.c from the downloaded archive")
                    endif()
                else()
                    message(STATUS "'make' not found; cannot build sqlite amalgamation from source")
                endif()
            endif()
        else()
            message(STATUS "Download attempt failed (${_dl_status}) for ${_try_url}")
        endif()
    endforeach()

    if(NOT _sqlite_amalg_ok)
        # Last resort: try FetchContent with the configured URL (may still fail)
        include(FetchContent)
        message(STATUS "Vendored sqlite amalgamation not found in source tree or download attempts; trying FetchContent from ${SQLITE_FETCH_URL}")
        FetchContent_Declare(
            sqlite_amalgamation
            URL ${SQLITE_FETCH_URL}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        # Prefer the modern helper; allow it to error out like before if unreachable
        FetchContent_MakeAvailable(sqlite_amalgamation)

        if(DEFINED sqlite_amalgamation_SOURCE_DIR)
            if(EXISTS "${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c")
                set(SQLITE_SRC "${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c")
                set(SQLITE_HDR "${sqlite_amalgamation_SOURCE_DIR}/sqlite3.h")
                set(VENDORED_SQLITE_DIR "${sqlite_amalgamation_SOURCE_DIR}")
            elseif(EXISTS "${sqlite_amalgamation_SOURCE_DIR}/tsrc/sqlite3.c")
                set(SQLITE_SRC "${sqlite_amalgamation_SOURCE_DIR}/tsrc/sqlite3.c")
                set(SQLITE_HDR "${sqlite_amalgamation_SOURCE_DIR}/tsrc/sqlite3.h")
                set(VENDORED_SQLITE_DIR "${sqlite_amalgamation_SOURCE_DIR}/tsrc")
            elseif(EXISTS "${sqlite_amalgamation_SOURCE_DIR}/sqlite/sqlite3.c")
                set(SQLITE_SRC "${sqlite_amalgamation_SOURCE_DIR}/sqlite/sqlite3.c")
                set(SQLITE_HDR "${sqlite_amalgamation_SOURCE_DIR}/sqlite/sqlite3.h")
                set(VENDORED_SQLITE_DIR "${sqlite_amalgamation_SOURCE_DIR}/sqlite")
            else()
                message(WARNING "Could not find sqlite3.c in fetched content; falling back to existing check (if any)")
            endif()
        endif()
    endif()
endif()

if(EXISTS ${SQLITE_SRC} AND EXISTS ${SQLITE_HDR})
    add_library(sqlite3_vendored STATIC ${SQLITE_SRC})
    target_include_directories(sqlite3_vendored PUBLIC ${VENDORED_SQLITE_DIR})

    # Compile-time options to harden SQLite and avoid dynamic extension loading
    target_compile_definitions(sqlite3_vendored PRIVATE SQLITE_OMIT_LOAD_EXTENSION=1 SQLITE_THREADSAFE=1)

    # If address sanitizer is enabled for this build, propagate ASan flags
    if(NN_ENABLE_ASAN)
        message(STATUS "Building vendored SQLite with ASan flags")
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(sqlite3_vendored PRIVATE -fsanitize=address -fno-omit-frame-pointer)
            target_link_options(sqlite3_vendored PRIVATE -fsanitize=address)
        endif()
    endif()

    # Prefer this target by providing the canonical imported alias name.
    # New projects should use SQLite3::SQLite3.
    add_library(SQLite3::SQLite3 ALIAS sqlite3_vendored)
else()
    message(STATUS "USE_VENDORED_SQLITE=ON but ${SQLITE_SRC} not found; skip vendored sqlite")
endif()

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
        message(STATUS "Found system SQLite3 (preferred). Using system package")
        return()
    else()
        message(STATUS "System SQLite3 not found; will attempt vendored SQLite if available")
    endif()
else()
    message(STATUS "USE_VENDORED_SQLITE=ON: forcing vendored SQLite build")
endif()

# Optional: allow FetchContent download if amalgamation isn't checked in.
# Set `SQLITE_FETCH_URL` to override the upstream archive URL if desired.
if(NOT DEFINED SQLITE_FETCH_URL)
    set(SQLITE_FETCH_URL "https://github.com/sqlite/sqlite/archive/refs/tags/version-3.53.0.tar.gz")
endif()

# If the amalgamation is missing, try to fetch it at configure time.
if(NOT (EXISTS ${SQLITE_SRC} AND EXISTS ${SQLITE_HDR}))
    include(FetchContent)
    message(STATUS "Vendored sqlite amalgamation not found; attempting FetchContent from ${SQLITE_FETCH_URL}")
    FetchContent_Declare(
        sqlite_amalgamation
        URL ${SQLITE_FETCH_URL}
    )
    FetchContent_GetProperties(sqlite_amalgamation)
    if(NOT sqlite_amalgamation_POPULATED)
        FetchContent_Populate(sqlite_amalgamation)
    endif()

    # Try a few plausible locations for the generated sqlite3.c / sqlite3.h
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

    # Prefer this target by providing an imported alias name used in targets
    add_library(SQLite::SQLite3 ALIAS sqlite3_vendored)
else()
    message(STATUS "USE_VENDORED_SQLITE=ON but ${SQLITE_SRC} not found; skip vendored sqlite")
endif()

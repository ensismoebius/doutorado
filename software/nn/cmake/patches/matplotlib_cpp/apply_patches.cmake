# Apply_patches.cmake
#
# Runs as the FetchContent PATCH_COMMAND for matplotlib-cpp (cwd = source dir).
# 1. Replaces the upstream CMakeLists.txt with the minimal one in this dir.
# 2. Guards the `long long` NumPy type specializations in matplotlibcpp.h.
#
# Reason for (2): on Apple platforms <cstdint> defines int64_t/uint64_t as
# `long long` (not `long`), so the unconditional
# `select_npy_type<long long>` / `<unsigned long long>` specializations collide
# with the `int64_t`/`uint64_t` ones defined just above. libstdc++ (Linux)
# defines int64_t as `long`, so there is no collision there and the extra
# specializations are harmless.

if(NOT MATPLOTLIBCPP_SOURCE_DIR)
    message(FATAL_ERROR "MATPLOTLIBCPP_SOURCE_DIR must be provided via -D")
endif()

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt"
     DESTINATION "${MATPLOTLIBCPP_SOURCE_DIR}")

file(READ "${MATPLOTLIBCPP_SOURCE_DIR}/matplotlibcpp.h" _mpcpp_h)

string(REPLACE
"static_assert(sizeof(long long) == 8);
template <> struct select_npy_type<long long> { const static NPY_TYPES type = NPY_INT64; };
static_assert(sizeof(unsigned long long) == 8);
template <> struct select_npy_type<unsigned long long> { const static NPY_TYPES type = NPY_UINT64; };"
"#if !defined(__APPLE__)
static_assert(sizeof(long long) == 8);
template <> struct select_npy_type<long long> { const static NPY_TYPES type = NPY_INT64; };
static_assert(sizeof(unsigned long long) == 8);
template <> struct select_npy_type<unsigned long long> { const static NPY_TYPES type = NPY_UINT64; };
#endif"
_mpcpp_h "${_mpcpp_h}")

file(WRITE "${MATPLOTLIBCPP_SOURCE_DIR}/matplotlibcpp.h" "${_mpcpp_h}")

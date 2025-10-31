## ProjectSetup.cmake
# Sets up the flags for CMake and compiler (top-level defines project())

# Sets up the flags for CMake and compiler
include(cmake/Flags.cmake)

# Ensure our custom modules are visible to find_package() so we can provide
# a lightweight FindMATIO that prefers the vendored copy or an existing
# in-tree target. This must be set before matio-cpp does find_package(MATIO).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

enable_testing()
include(CTest)

# Prevent vendored subprojects from enabling their own tests during configure
# so we don't require test frameworks for third-party code while configuring.
set(BUILD_TESTING OFF CACHE BOOL "Disable building tests in subprojects" FORCE)

# Define paths for convenience
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")

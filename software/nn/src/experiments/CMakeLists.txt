# Sanitizer flags for Debug builds (NN_ENABLE_ASAN), applied to every experiment below.
include(../../cmake/SanitizerFlags.cmake)

# One folder per experiment; each defines its library, executable, tests and profiles.
# meeting01 comes first: thesis_lib links meeting01_lib.
add_subdirectory(meeting01)
add_subdirectory(thesis)
add_subdirectory(paraconsistentGA)

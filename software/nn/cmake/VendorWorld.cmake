##
## VendorWorld.cmake
##
## Purpose
## - Fetch WORLD (mmorise/World), the vocoder the thesis uses to edit a voice's F0 and
##   envelope for simulated dysphonia (VoiceDysphonia.hpp): Harvest for F0, CheapTrick for the
##   spectral envelope, D4C for the aperiodicity, Synthesis() to rebuild the signal.
##
## What it provides
## - Target: `world_vendor` (STATIC), built from the library sources only.
## - Include path: `${world_SOURCE_DIR}/src`, headers included as <world/synthesis.h>.
##
## Local policy
## - Licence: modified BSD (three clauses: keep the notice, no endorsement); the README states
##   "no patent in all algorithms". The licence text ships in `${world_SOURCE_DIR}/LICENSE.txt`.
##   Citation the README asks for: "WORLD [1] (D4C edition [2])", i.e. Morise, Yokomori, Ozawa
##   (2016) IEICE Trans. Inf. Syst. E99-D(7) and Morise (2016) Speech Communication 84:57-65.
## - WORLD's own CMakeLists.txt is NOT used: it writes into ${CMAKE_BINARY_DIR}/lib and /bin,
##   installs and exports targets, and builds examples. Only the sources are compiled here.
## - Pinned to a commit for reproducibility (2026-10-09, "Update README.md").
## - No mutable global state in the library: each call that needs random numbers owns a
##   RandnState seeded by randn_reseed(), so concurrent calls (OpenMP) do not interact. This is
##   also checked by a test (VoiceDysphonia thread independence), not only read from the code.
##

include(FetchContent)

FetchContent_Declare(
    world
    GIT_REPOSITORY https://github.com/mmorise/World.git
    GIT_TAG        c794856b537758a3d300757db2e5e31685a822c2
    # A SOURCE_SUBDIR without a CMakeLists.txt: FetchContent downloads the tree but does not
    # add_subdirectory() it (see the policy above).
    SOURCE_SUBDIR  cmake-not-used
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(world)

add_library(world_vendor STATIC
    ${world_SOURCE_DIR}/src/cheaptrick.cpp
    ${world_SOURCE_DIR}/src/codec.cpp
    ${world_SOURCE_DIR}/src/common.cpp
    ${world_SOURCE_DIR}/src/d4c.cpp
    ${world_SOURCE_DIR}/src/dio.cpp
    ${world_SOURCE_DIR}/src/fft.cpp
    ${world_SOURCE_DIR}/src/harvest.cpp
    ${world_SOURCE_DIR}/src/matlabfunctions.cpp
    ${world_SOURCE_DIR}/src/stonemask.cpp
    ${world_SOURCE_DIR}/src/synthesis.cpp
    ${world_SOURCE_DIR}/src/synthesisrealtime.cpp
)
target_include_directories(world_vendor SYSTEM PUBLIC "$<BUILD_INTERFACE:${world_SOURCE_DIR}/src>")
target_compile_options(world_vendor PRIVATE -w -fno-lto)
set_property(TARGET world_vendor PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
set_target_properties(world_vendor PROPERTIES CXX_CLANG_TIDY "")

# Fetch and make available googletest
include(FetchContent)

message(STATUS "Configuring GTest...")

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE
)

FetchContent_MakeAvailable(googletest)

# ---------------------------------------------
# Create a modular wrapper target
# ---------------------------------------------
add_library(google_test INTERFACE)

# Link to GoogleTest
target_link_libraries(google_test
    INTERFACE
        GTest::gtest
        GTest::gtest_main
)

# Suppress warnings and clang-tidy for GTest targets
if(TARGET gtest)
    target_compile_options(gtest PRIVATE -w)
    set_target_properties(gtest PROPERTIES CXX_CLANG_TIDY "")
endif()

if(TARGET gtest_main)
    target_compile_options(gtest_main PRIVATE -w)
    set_target_properties(gtest_main PROPERTIES CXX_CLANG_TIDY "")
endif()

# Extract GoogleTest include dirs
get_target_property(
    GTEST_INCLUDES
    GTest::gtest
    INTERFACE_INCLUDE_DIRECTORIES
)

if(GTEST_INCLUDES)
    if(TARGET gtest)
        target_include_directories(gtest
            SYSTEM INTERFACE
                ${GTEST_INCLUDES}
        )
    endif()
endif()

get_target_property(
    GTEST_MAIN_INCLUDES
    GTest::gtest_main
    INTERFACE_INCLUDE_DIRECTORIES
)

if(GTEST_MAIN_INCLUDES)
    if(TARGET gtest_main)
        target_include_directories(gtest_main
            SYSTEM INTERFACE
                ${GTEST_MAIN_INCLUDES}
        )
    endif()
endif()

# Re-expose them as SYSTEM
target_include_directories(google_test
    SYSTEM INTERFACE
        ${GTEST_INCLUDES}
)


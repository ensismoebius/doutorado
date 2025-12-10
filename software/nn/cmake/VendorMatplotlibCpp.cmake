# VendorMatplotlibCpp.cmake
# Configure vendored matplotlib-cpp presence and make MatplotlibCpp::MatplotlibCpp available

include(ExternalProject)

# Set install directory
set(MATPLOTLIBCPP_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/matplotlib-cpp-install")

# Use ExternalProject to fetch and install matplotlib-cpp
ExternalProject_Add(matplotlib-cpp
    GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
    GIT_TAG        ef0383f1315d32e0156335e10b82e90b334f6d9f
    SOURCE_DIR     "${CMAKE_BINARY_DIR}/_deps/matplotlib-cpp-src"
    INSTALL_DIR    "${MATPLOTLIBCPP_INSTALL_DIR}"
    CONFIGURE_COMMAND ""  # matplotlib-cpp is header-only, no configure step
    BUILD_COMMAND   ""    # matplotlib-cpp is header-only, no build step
    INSTALL_COMMAND ""    # matplotlib-cpp is header-only, no install step needed by EP
    UPDATE_DISCONNECTED 1
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Create an imported interface library for matplotlib-cpp
add_library(MatplotlibCpp::MatplotlibCpp INTERFACE IMPORTED)

# Add a dependency on the external project to ensure it's fetched
add_dependencies(MatplotlibCpp::MatplotlibCpp matplotlib-cpp)

# Use FindPython to locate Python interpreter and development headers
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Need to ensure that Python headers are available for matplotlib-cpp
if(Python3_FOUND)
    message(STATUS "Found Python for Matplotlib-cpp: ${Python3_LIBRARIES}")
    
    set_target_properties(MatplotlibCpp::MatplotlibCpp PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/_deps/matplotlib-cpp-src;${Python3_INCLUDE_DIRS};${Python3_NumPy_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${Python3_LIBRARIES}"
    )
else()
    message(FATAL_ERROR "Python not found, matplotlib-cpp requires Python development headers.")
endif()
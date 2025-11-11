cmake_minimum_required(VERSION 3.10)

# Project declaration
project(
		nn
		VERSION 0.1.0
		LANGUAGES CXX)

# --- CMake policy settings ---
# Prefer modern behavior for these policies. Use guarded checks so the file
# remains compatible with older CMake versions that don't know about these
# policy names.
if(POLICY CMP0156)
	# De-duplicate libraries on link lines based on linker capabilities
	cmake_policy(SET CMP0156 NEW)
endif()
if(POLICY CMP0181)
	# Parse and re-quote link command-line fragment variables correctly
	cmake_policy(SET CMP0181 NEW)
endif()
if(POLICY CMP0082)
	# Avoid interleaving install rules from add_subdirectory() with caller
	cmake_policy(SET CMP0082 NEW)
endif()


## Top-level CMake: delegate to modular includes in cmake/

include(cmake/ProjectSetup.cmake)

# Do the required libraries check (package checks remain small and local)
include(cmake/PackageChecking.cmake)

# Vendor includes and helpers (cnpy, imgui, implot, eigen parallel settings)
include(cmake/VendorIncludes.cmake)

# Vendored matio handling (sets MATIO_ROOT_DIR, MATIO_INCLUDE_DIR, MATIO_LIBRARY, exposes MATIO::MATIO)
include(cmake/VendorMatio.cmake)

# Workarounds / shims for matio-cpp configure/export behavior
include(cmake/VendorMatioCppShim.cmake)

# Main wiring
include(cmake/Main.cmake)

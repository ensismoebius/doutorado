#+#+#+#+-----------------------------------------------------------------------
# Policies.cmake
#
# Centralizes CMake policy selection so the rest of the build can assume modern,
# consistent behavior across CMake versions.
#+#+#+#+-----------------------------------------------------------------------

# --- CMake policy settings ---
# Prefer modern behavior for these policies. Use guarded checks so the file
# remains compatible with older CMake versions that don't know about these
# policy names.

if(POLICY CMP0135)
	# Set the policy for timestamp handling in FetchContent
	cmake_policy(SET CMP0135 NEW)
endif()

if(POLICY CMP0156)
	# De-duplicate libraries on link lines based on linker capabilities
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0156.html
	cmake_policy(SET CMP0156 NEW)
endif()
if(POLICY CMP0181)
	# Parse and re-quote link command-line fragment variables correctly
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0181.html
	cmake_policy(SET CMP0181 NEW)
endif()
if(POLICY CMP0082)
	# Avoid interleaving install rules from add_subdirectory() with caller
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0082.html
	cmake_policy(SET CMP0082 NEW)
endif()
if(POLICY CMP0003)
	# Use the target name as the link library name avoiding legacy behavior
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0003.html
    cmake_policy(SET CMP0003 NEW)
endif()
if(POLICY CMP0048)
	# Use project() version for PACKAGE_VERSION, demanding explicit versioning
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0048.html
    cmake_policy(SET CMP0048 NEW)
endif()
if(POLICY CMP0057)
	# Allow if() conditions to use target names directly
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0057.html
    cmake_policy(SET CMP0057 NEW)
endif()
if(POLICY CMP0063)
	# Honor visibility properties for all target types
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0063.html
    cmake_policy(SET CMP0063 NEW)
endif()
if(POLICY CMP0077)
	# Allow option() to modify cached variables, enabling better configurability
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0077.html
    cmake_policy(SET CMP0077 NEW)
endif()
if(POLICY CMP0135)
	# Avoid non-determinist timestamp in generated files
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0135.html
    cmake_policy(SET CMP0135 NEW)
endif()
if(POLICY CMP0167)
	# Refine handling of INTERFACE_LINK_LIBRARIES for OBJECT libraries
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0167.html
    cmake_policy(SET CMP0167 NEW)
endif()
if(POLICY CMP0169)
	# Improve handling of target names in generator expressions
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0169.html
    cmake_policy(SET CMP0169 NEW)
endif()

if(POLICY CMP0069)
	# Prefer the new behavior for handling link options and the
	# availability of add_link_options(). Centralized here so other
	# cmake modules can assume consistent policy configuration.
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0069.html
	cmake_policy(SET CMP0069 NEW)
endif()

if(POLICY CMP0079)
	# Allow target_link_libraries() to be used with targets defined in other 
	# directories than the current one.
	# Explanation: https://cmake.org/cmake/help/latest/policy/CMP0079.html
	cmake_policy(SET CMP0079 NEW)
endif()
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
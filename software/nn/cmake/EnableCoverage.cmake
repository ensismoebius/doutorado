option(NN_ENABLE_COVERAGE "Enable coverage instrumentation (adds --coverage flags)" OFF)

if(NN_ENABLE_COVERAGE)
  if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(STATUS "NN_ENABLE_COVERAGE=ON: adding coverage compile/link flags")
    # Keep existing flags and append coverage instrumentation flags used by lcov/gcov
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -O0 -g -fno-inline")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -O0 -g -fno-inline")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  else()
    message(WARNING "NN_ENABLE_COVERAGE set but compiler is not GCC/Clang; coverage flags not added.")
  endif()
endif()

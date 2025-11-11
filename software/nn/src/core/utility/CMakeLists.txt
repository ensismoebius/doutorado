add_library(utility STATIC
    comparison.cpp
)

set_property(TARGET utility PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(utility
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
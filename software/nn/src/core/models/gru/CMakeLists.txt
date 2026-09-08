# Core GRU models library
# Headers live in include/models/gru/; implementation here.

add_library(gru_models STATIC
    GRUAutoencoder.cpp
)

target_include_directories(gru_models
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(gru_models
    PUBLIC
        tensor
        layers
)

set_property(TARGET gru_models PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(tests)

# Core Transformer models library
# Headers live in include/models/transformer/ and include/layers/attention/.

add_library(transformer_models STATIC
    TransformerAutoencoder.cpp
)

target_include_directories(transformer_models
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(transformer_models
    PUBLIC
        tensor
        layers
)

set_property(TARGET transformer_models PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(tests)

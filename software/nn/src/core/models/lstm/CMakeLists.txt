# Core LSTM models library
# Headers live in include/models/lstm/; implementation here.

add_library(lstm_models STATIC
    LSTMAutoencoder.cpp
)

target_include_directories(lstm_models
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(lstm_models
    PUBLIC
        tensor
        layers
)

set_property(TARGET lstm_models PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(tests)
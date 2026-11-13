# Core autoencoder models library
# Headers live in include/models/autoencoder/; implementation here.
# The thesis uses ProtocolAutoencoder (ANN-AE) and ProtocolSpikingAutoencoder (SNN-AE);
# meeting01 uses AutoencoderConfig and the builders.

add_library(autoencoder_models STATIC
    EncoderDecoderAutoencoder.cpp
    AudioWindowAutoencoder.cpp
    AudioWindowSpikingAutoencoder.cpp
    EegWindowAutoencoder.cpp
    EegWindowSpikingAutoencoder.cpp
    FusedWindowAutoencoder.cpp
    FusedWindowSpikingAutoencoder.cpp
    ProtocolAutoencoder.cpp
    ProtocolSpikingAutoencoder.cpp
)

target_include_directories(autoencoder_models
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(autoencoder_models
    PUBLIC
        tensor
        layers
)

set_property(TARGET autoencoder_models PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(tests)

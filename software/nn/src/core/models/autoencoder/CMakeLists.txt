# Core autoencoder models library
# Headers live in include/models/autoencoder/; implementation here.
#
# These models were physically inside src/experiments/autoencoderRunner/lib
# while THREE experiments consumed them (autoencoderRunner, guayaquil,
# thesis), so guayaquil and thesis had to depend on another experiment's
# include directories to build. They are core models; this is where they
# belong. The dead parallel design that used to sit here -- headers no
# translation unit ever read -- was deleted in the same change.

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

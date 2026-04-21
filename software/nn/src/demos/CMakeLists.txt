# Add sanitizer flags
include(../../cmake/SanitizerFlags.cmake)

# Add subdirectories for all demos and experiments
add_subdirectory(cppdemos)
add_subdirectory(fftw3_demo)
# add_subdirectory(autoencoder)  # Moved to tests/autoencoder_leakyrelu
add_subdirectory(wavelet_demo)
add_subdirectory(lfcc_pipeline)
add_subdirectory(exec_resnet_demo)
add_subdirectory(voice_biometrics_cpp)
add_subdirectory(exec_plotSpikingNetwork)
add_subdirectory(tests/autoencoder_leakyrelu)

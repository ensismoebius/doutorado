# Add sanitizer flags
include(../../cmake/SanitizerFlags.cmake)

# Add subdirectories for all demos and experiments
add_subdirectory(cppDemos/speaker_demo)
add_subdirectory(cppDemos/fftw3_demo)
add_subdirectory(cppDemos/wavelet_demo)
add_subdirectory(cppDemos/lfcc_pipeline)
add_subdirectory(cppDemos/exec_resnet_demo)
add_subdirectory(cppDemos/voice_biometrics_cpp)
add_subdirectory(cppDemos/exec_plotSpikingNetwork)
add_subdirectory(cppDemos/tests/autoencoder_leakyrelu)

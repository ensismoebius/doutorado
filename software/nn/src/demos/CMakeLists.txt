# Add sanitizer flags
include(../../cmake/SanitizerFlags.cmake)

# Add subdirectories for all demos and experiments
add_subdirectory(cppdemos)
add_subdirectory(fftw3_demo)
add_subdirectory(wavelet_demo)
add_subdirectory(lfcc_pipeline)
add_subdirectory(exec_resnet_demo)
add_subdirectory(exec_loadingData)
add_subdirectory(exec_plotSpikingNetwork)
add_subdirectory(autoEncoderLeakyReLUTest)
add_subdirectory(autoEncoderLeakyReLUAndSpikeTest)

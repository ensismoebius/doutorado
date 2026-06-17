# Add sanitizer flags
include(../../cmake/SanitizerFlags.cmake)

# Add subdirectories for all demos and experiments
add_subdirectory(cppDemos/snn_speaker_demo)
add_subdirectory(cppDemos/fft_demo)
add_subdirectory(cppDemos/wavelet_demo)
add_subdirectory(cppDemos/lfcc_feature_demo)
add_subdirectory(cppDemos/resnet_classifier_demo)
add_subdirectory(cppDemos/wpt_voice_biometrics)
add_subdirectory(cppDemos/snn_spike_plotter)
add_subdirectory(cppDemos/autoencoder_leakyrelu)

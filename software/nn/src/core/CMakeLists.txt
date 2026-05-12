# Add sanitizer flags
include(../../cmake/SanitizerFlags.cmake)

# Add subdirectories for all modules
add_subdirectory(wave)
add_subdirectory(serialization)
add_subdirectory(layers)
add_subdirectory(tensor)
add_subdirectory(wavelet)
add_subdirectory(utility)
add_subdirectory(statistics)
add_subdirectory(optimizers)
add_subdirectory(data_loaders)
add_subdirectory(initializers)
add_subdirectory(linear_algebra)
add_subdirectory(paraconsistent)
add_subdirectory(models)
add_subdirectory(training)
add_subdirectory(progress)


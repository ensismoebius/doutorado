# Add Armadillo and OpenMP dependencies
# find_package(Armadillo REQUIRED)
find_package(OpenMP REQUIRED)

# Add OpenGL dependencies
find_package(OpenGL REQUIRED)

# Find SDL2
find_package(SDL2 REQUIRED)

# Package finder
find_package(PkgConfig REQUIRED)

# Find ALSA library
pkg_check_modules(ASOUND REQUIRED alsa)

# Find GLFW package
pkg_check_modules(GLFW REQUIRED glfw3)

# Find PortAudio
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
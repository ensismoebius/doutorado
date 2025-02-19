#include <raylib.h>
#include <lsl_cpp.h>
#include <alsa/asoundlib.h>

#include <vector>
#include <iostream>
#include <thread>
#include <stdexcept>

#include "Window.cpp"

#include "../lib/widgets/CustomImGuiWindow.cpp"

// Define constants
const int screenWidth = 800;
const int screenHeight = 600;

const int channels = 1;
const int bufferSize = 256;
unsigned sampleRate = 44100;

// Data container
float audioData[bufferSize];

std::string title = "Teste";
ImVec2 dimensions = {800, 600};
ImVec4 color = {255, 255, 255, 255};

Window app(title, dimensions, color);

// Main function
int main()
{
    app.run();
    return 0;
}

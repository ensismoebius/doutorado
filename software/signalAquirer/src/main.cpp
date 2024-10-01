#include <raylib.h>
#include <alsa/asoundlib.h>

#include <vector>
#include <iostream>
#include <thread>

#include "Window.cpp"

#include "../lib/capture/AudioCapturer.h"
#include "../lib/widgets/CustomImGuiWindow.cpp"

// Define constants
const int screenWidth = 800;
const int screenHeight = 600; // Increased height to fit the button

const int bufferSize = 4096;
const int channels = 1; // Mono
unsigned sampleRate = 44100;

int samplingStep = 1;             // Default downsampling rate
bool downsamplingEnabled = false; // State for the toggle button

// Data container
float audioData[bufferSize];

AudioCapturer capturer(sampleRate, channels);

// Using std::jthread in C++20 (automatically joins on destruction)
void threadCaptureAudioData(std::stop_token stopToken)
{
    while (!stopToken.stop_requested())
    {
        // Read audio data
        capturer.captureAudio(audioData, bufferSize);
    }
    std::cout << "Thread is stopping!" << std::endl;
}

bool teste = false;

std::string title = "Teste";
ImVec2 dimensions = {800, 600};
ImVec4 color = {255, 255, 255, 255};

Window app(title, dimensions, color);

// Main function
int main()
{
    std::jthread jt(threadCaptureAudioData);
    app.data = audioData;
    app.run();

    jt.request_stop();
    return 0;
}

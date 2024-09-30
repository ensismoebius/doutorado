#include <raylib.h>
#include <alsa/asoundlib.h>

#include <vector>
#include <iostream>
#include <thread>

#include "Window.cpp"
#include "../lib/widgets/ScrollPanel.h"
#include "../lib/capture/AudioCapturer.h"

// Define constants
const int screenWidth = 800;
const int screenHeight = 600; // Increased height to fit the button

const int bufferSize = 4096;
const int channels = 1; // Mono
unsigned sampleRate = 44100;

int samplingStep = 1;             // Default downsampling rate
bool downsamplingEnabled = false; // State for the toggle button

// Data container
std::vector<short> audioData(bufferSize);

AudioCapturer capturer(sampleRate, channels);

// Using std::jthread in C++20 (automatically joins on destruction)
void threadCaptureAudioData(std::stop_token stopToken)
{
    while (!stopToken.stop_requested())
    {
        // Read audio data
        capturer.captureAudio(&audioData, bufferSize);
    }
    std::cout << "Thread is stopping!" << std::endl;
}

bool teste = false;

Window app("Teste", ImVec2(800, 600), ImVec4(255, 255, 255, 255));

// Main function
int main()
{

    std::jthread jt(threadCaptureAudioData);

    jt.request_stop();
    return 0;
}

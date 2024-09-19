#include <raylib.h>
#include <raygui.h>
#include <alsa/asoundlib.h>
#include <vector>
#include <iostream>
#include <thread>

#include "../lib/capture/AudioCapturer.h"

// Define constants
const int screenWidth = 640;
const int screenHeight = 420; // Increased height to fit the button

const int bufferSize = 4096;
const int channels = 1; // Mono
unsigned sampleRate = 44100;

// Data conteiner
std::vector<short> audioData(bufferSize);

// Ploting points
std::vector<Vector2> plottingPoints(screenWidth);

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

// Function for downsampling the audio data for better performance
void downsampleAudioData(int downsampleRate, int screenWidth, int screenHeight)
{
    float step = (float)(audioData.size() / downsampleRate) / screenWidth;

    for (int i = 0; i < screenWidth; ++i)
    {
        int index = (int)(i * step);
        plottingPoints[i] = Vector2{(float)i, screenHeight / 2 - audioData[index] / 256.0f}; // Scale down for visualization
    }

    // Draw the audio plot
    if (screenWidth > 1)
    {
        DrawLineStrip(plottingPoints.data(), screenWidth, BLUE);
    }
}

// Function for generating raw audio points (no downsampling)
void generateRawAudioPoints(int screenWidth, int screenHeight)
{
    static int startPosX, startPosY, endPosX, endPosY;

    std::vector<Vector2> points(screenWidth);
    float step = (float)audioData.size() / screenWidth;

    for (int i = 0; i < screenWidth; ++i)
    {
        int index = (int)(i * step);
        points[i] = Vector2{(float)i, screenHeight / 2 - audioData[index] / 128.0f}; // Scale down for visualization
    }

    // Draw the audio plot
    if (screenWidth > 1)
    {
        DrawLineStrip(points.data(), screenWidth, BLUE);
    }
}

// Main function
int main()
{

    // Using std::jthread in C++20 (automatically joins on destruction)
    std::jthread jt(threadCaptureAudioData);

    // Initialize raylib
    InitWindow(screenWidth, screenHeight, "Audio Plot with Downsampling Toggle");
    SetTargetFPS(60);

    int downsampleRate = 64;         // Default downsampling rate
    bool downsamplingEnabled = true; // State for the toggle button

    // Main loop
    while (!WindowShouldClose())
    {

        // Clear the screen
        BeginDrawing();
        ClearBackground(BLACK);

        if (downsamplingEnabled)
        {
            downsampleAudioData(downsampleRate, screenWidth, screenHeight);
        }
        else
        {
            generateRawAudioPoints(screenWidth, screenHeight);
        }

        // Draw the toggle button
        if (GuiButton((Rectangle){screenWidth / 2 - 50, screenHeight - 40, 100, 30}, downsamplingEnabled ? "Downsample: ON" : "Downsample: OFF"))
        {
            downsamplingEnabled = !downsamplingEnabled; // Toggle state on button press
        }

        EndDrawing();
    }

    // Clean up
    CloseWindow();

    jt.request_stop();

    return 0;
}

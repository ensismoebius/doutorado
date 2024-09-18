#include <raylib.h>
#include <raygui.h> // Required for GUI buttons
#include <alsa/asoundlib.h>
#include <vector>
#include <iostream>

#include "../lib/AudioCapturer.h"

// Define constants
const int screenWidth = 1024;
const int screenHeight = 420; // Increased height to fit the button

const int bufferSize = 4096;
const int channels = 1; // Mono
unsigned sampleRate = 44100;

// Function for downsampling the audio data for better performance
std::vector<Vector2> downsampleAudioData(const std::vector<short> &audioData, int numPoints, int downsampleRate)
{
    std::vector<Vector2> points(numPoints);
    float step = (float)(audioData.size() / downsampleRate) / numPoints;

    for (int i = 0; i < numPoints; ++i)
    {
        int index = (int)(i * step);
        points[i] = Vector2{(float)i, screenHeight / 2 - audioData[index] / 256.0f}; // Scale down for visualization
    }
    return points;
}

// Function for generating raw audio points (no downsampling)
std::vector<Vector2> generateRawAudioPoints(const std::vector<short> &audioData, int numPoints)
{
    std::vector<Vector2> points(numPoints);
    float step = (float)audioData.size() / numPoints;

    for (int i = 0; i < numPoints; ++i)
    {
        int index = (int)(i * step);
        points[i] = Vector2{(float)i, screenHeight / 2 - audioData[index] / 256.0f}; // Scale down for visualization
    }
    return points;
}

// Main function
int main()
{
    // Initialize raylib
    InitWindow(screenWidth, screenHeight, "Audio Plot with Downsampling Toggle");
    SetTargetFPS(60);

    AudioCapturer capturer(sampleRate, channels);

    int downsampleRate = 64;         // Default downsampling rate
    bool downsamplingEnabled = true; // State for the toggle button

    // Main loop
    while (!WindowShouldClose())
    {
        // Read audio data
        std::vector<short> audioData = capturer.captureAudio(bufferSize);

        // Downsample or use raw data based on toggle state
        int numPoints = screenWidth; // Only plot one point per pixel
        std::vector<Vector2> points;
        if (downsamplingEnabled)
        {
            points = downsampleAudioData(audioData, numPoints, downsampleRate);
        }
        else
        {
            points = generateRawAudioPoints(audioData, numPoints);
        }

        // Clear the screen
        BeginDrawing();
        ClearBackground(BLACK);

        // Draw the audio plot
        if (numPoints > 1)
        {
            DrawLineStrip(points.data(), numPoints, BLUE);
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

    return 0;
}

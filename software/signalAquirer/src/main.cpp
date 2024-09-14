#include <raylib.h>
#include <alsa/asoundlib.h>
#include <vector>
#include <iostream>

// Define constants
const int screenWidth = 800;
const int screenHeight = 600;
const int bufferSize = 1024;
const int sampleRate = 44100;

// Function to initialize ALSA
snd_pcm_t *initAlsa()
{
    snd_pcm_t *handle;
    snd_pcm_hw_params_t *params;
    unsigned int rate = sampleRate;
    int dir;
    snd_pcm_uframes_t frames = bufferSize;
    int rc;

    rc = snd_pcm_open(&handle, "default", SND_PCM_STREAM_CAPTURE, 0);
    if (rc < 0)
    {
        std::cerr << "Unable to open PCM device: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(handle, params);
    snd_pcm_hw_params_set_format(handle, params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_rate_near(handle, params, &rate, &dir);
    snd_pcm_hw_params_set_channels(handle, params, 1);
    snd_pcm_hw_params_set_period_size(handle, params, frames, 0);
    rc = snd_pcm_hw_params(handle, params);
    if (rc < 0)
    {
        std::cerr << "Unable to set HW params: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    return handle;
}

// Function to read audio data from ALSA
std::vector<short> readAudioData(snd_pcm_t *handle)
{
    std::vector<short> buffer(bufferSize);
    int rc = snd_pcm_readi(handle, buffer.data(), bufferSize);
    if (rc < 0)
    {
        std::cerr << "Error reading audio data: " << snd_strerror(rc) << std::endl;
        return std::vector<short>(bufferSize, 0); // Return zeroed buffer on error
    }
    else if (rc != bufferSize)
    {
        std::cerr << "Short read, read " << rc << " frames" << std::endl;
    }
    return buffer;
}

// Main function
int main()
{
    // Initialize raylib
    InitWindow(screenWidth, screenHeight, "Audio Plot with raylib");
    SetTargetFPS(60);

    // Initialize ALSA
    snd_pcm_t *alsaHandle = initAlsa();

    // Main loop
    while (!WindowShouldClose())
    {
        // Read audio data
        std::vector<short> audioData = readAudioData(alsaHandle);

        // Calculate the number of points in the plot
        int numPoints = screenWidth;
        float xStep = (float)bufferSize / numPoints;

        // Prepare points for plotting
        std::vector<Vector2> points(numPoints);
        for (int i = 0; i < numPoints; ++i)
        {
            int index = (int)(i * xStep);
            points[i] = Vector2{(float)i, screenHeight / 2 - audioData[index] / 256.0f};
        }

        // Clear the screen
        BeginDrawing();
        ClearBackground(RAYWHITE);

        // Draw the audio plot
        if (numPoints > 1)
        {
            DrawLineStrip(points.data(), numPoints, BLACK);
        }

        EndDrawing();
    }

    // Clean up
    snd_pcm_close(alsaHandle);
    CloseWindow();

    return 0;
}

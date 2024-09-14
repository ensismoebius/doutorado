#include <raylib.h>
#include <alsa/asoundlib.h>
#include <vector>
#include <iostream>

// Define constants
const int screenWidth = 800;
const int screenHeight = 600;
const int bufferSize = 1024;
const int sampleRate = 44100;

snd_pcm_t *initAlsa()
{
    snd_pcm_t *handle;
    snd_pcm_hw_params_t *params;
    unsigned int rate = sampleRate;
    int dir;
    snd_pcm_uframes_t frames = bufferSize;
    int rc;
    snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE; // Default format

    // Open PCM device
    rc = snd_pcm_open(&handle, "default", SND_PCM_STREAM_CAPTURE, 0);
    if (rc < 0)
    {
        std::cerr << "Unable to open PCM device: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(handle, params);

    // Test and select supported format
    snd_pcm_format_t formats[] = {SND_PCM_FORMAT_S16_LE, SND_PCM_FORMAT_S32_LE, SND_PCM_FORMAT_U8};
    bool formatSupported = false;

    for (snd_pcm_format_t fmt : formats)
    {
        rc = snd_pcm_hw_params_test_format(handle, params, fmt);
        if (rc == 0)
        {
            format = fmt;
            formatSupported = true;
            break;
        }
    }

    if (!formatSupported)
    {
        std::cerr << "No supported format found" << std::endl;
        snd_pcm_close(handle);
        exit(EXIT_FAILURE);
    }

    // Set format
    rc = snd_pcm_hw_params_set_format(handle, params, format);
    if (rc < 0)
    {
        std::cerr << "Unable to set format: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Set sample rate
    rc = snd_pcm_hw_params_set_rate_near(handle, params, &rate, &dir);
    if (rc < 0)
    {
        std::cerr << "Unable to set rate: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Set channels
    rc = snd_pcm_hw_params_set_channels(handle, params, 1);
    if (rc < 0)
    {
        std::cerr << "Unable to set channels: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Set period size
    rc = snd_pcm_hw_params_set_period_size(handle, params, frames, 0);
    if (rc < 0)
    {
        std::cerr << "Unable to set period size: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Apply hardware parameters
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

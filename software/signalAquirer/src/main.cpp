#include <raylib.h>
#include <raygui.h>  // Required for GUI buttons
#include <alsa/asoundlib.h>
#include <vector>
#include <iostream>

// Define constants
const int screenWidth = 1024;
const int screenHeight = 420; // Increased height to fit the button
const int bufferSize = 4096;
const int channels = 1; // Mono
unsigned sampleRate = 44100;

snd_pcm_t *initAlsa()
{
    snd_pcm_t *handle;
    snd_pcm_hw_params_t *params;
    snd_pcm_sw_params_t *sw_params;
    int rc;

    snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;

    rc = snd_pcm_open(&handle, "default", SND_PCM_STREAM_CAPTURE, 0);
    if (rc < 0)
    {
        std::cerr << "Failed to open PCM device: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    snd_pcm_hw_params_alloca(&params);
    snd_pcm_sw_params_alloca(&sw_params);

    rc = snd_pcm_hw_params_any(handle, params);
    if (rc < 0)
    {
        std::cerr << "Failed to initialize hardware parameters: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_hw_params_set_access(handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (rc < 0)
    {
        std::cerr << "Failed to set access type: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_hw_params_set_format(handle, params, format);
    if (rc < 0)
    {
        std::cerr << "Unable to set format: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_hw_params_set_channels(handle, params, channels);
    if (rc < 0)
    {
        std::cerr << "Failed to set channels: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_hw_params_set_rate_near(handle, params, &sampleRate, 0);
    if (rc < 0)
    {
        std::cerr << "Failed to set rate: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_hw_params(handle, params);
    if (rc < 0)
    {
        std::cerr << "Failed to set hardware parameters: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_sw_params_current(handle, sw_params);
    if (rc < 0)
    {
        std::cerr << "Failed to get software parameters: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_sw_params_set_start_threshold(handle, sw_params, 0);
    if (rc < 0)
    {
        std::cerr << "Failed to set start threshold: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_sw_params(handle, sw_params);
    if (rc < 0)
    {
        std::cerr << "Failed to set software parameters: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    return handle;
}

// Function to read audio data from ALSA
std::vector<short> readAudioData(snd_pcm_t *handle)
{
    std::vector<short> buffer(bufferSize);
    int rc = snd_pcm_readi(handle, buffer.data(), bufferSize / channels);
    if (rc < 0)
    {
        std::cerr << "Error reading audio data: " << snd_strerror(rc) << std::endl;
        return std::vector<short>(bufferSize, 0); // Return zeroed buffer on error
    }
    return buffer;
}

// Function for downsampling the audio data for better performance
std::vector<Vector2> downsampleAudioData(const std::vector<short>& audioData, int numPoints, int downsampleRate)
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
std::vector<Vector2> generateRawAudioPoints(const std::vector<short>& audioData, int numPoints)
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

    // Initialize ALSA
    snd_pcm_t *alsaHandle = initAlsa();

    int downsampleRate = 2; // Default downsampling rate
    bool downsamplingEnabled = true; // State for the toggle button

    // Main loop
    while (!WindowShouldClose())
    {
        // Read audio data
        std::vector<short> audioData = readAudioData(alsaHandle);

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
        ClearBackground(RAYWHITE);

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
    snd_pcm_close(alsaHandle);
    CloseWindow();

    return 0;
}

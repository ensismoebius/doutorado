#include <raylib.h>
#include <raygui.h>
#include <rlgl.h>
#include <alsa/asoundlib.h>
#include <vector>
#include <iostream>
#include <thread>

#include "../lib/capture/AudioCapturer.h"

// Define constants
const int screenWidth = 800;
const int screenHeight = 600; // Increased height to fit the button

const int bufferSize = 4096;
const int channels = 1; // Mono
unsigned sampleRate = 44100;

int samplingStep = 1;             // Default downsampling rate
bool downsamplingEnabled = false; // State for the toggle button

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

void plotSignal(const int screenWidth, const int screenHeight, const int bufferSize, const int sampling = 1)
{
    // Reinicia as variáveis no início de cada chamada
    int startPosX = 0;
    int startPosY = screenHeight / 2; // Começar no meio da tela

    float step = (float)bufferSize / screenWidth / sampling;
    float index = 0.0f;

    rlBegin(RL_LINES);
    rlColor4ub(BLUE.r, BLUE.g, BLUE.b, BLUE.a);

    for (int i = 0; i < screenWidth; ++i)
    {
        // Calcula a nova posição baseada no dado de áudio
        int endPosX = i;
        int endPosY = screenHeight / 2 - audioData[index] / 128.0f;

        rlVertex2i(startPosX, startPosY);
        rlVertex2i(endPosX, endPosY);

        // Atualiza os pontos de início
        startPosX = endPosX;
        startPosY = endPosY;

        // Avança no áudio
        index += step;
    }

    rlEnd();
}

inline void drawUI()
{

    static float i = 0;
    // Variables for sine wave
    float offsetX = 0.0f;       // Horizontal offset for scrolling
    float maxOffsetX = 5000.0f; // Max value for horizontal scrolling

    DrawText("Wave is scrolling continuously", 10, screenHeight - 60, 20, DARKGRAY);

    // Manual control of scroll position using slider
    GuiSlider(Rectangle{10, screenHeight - 40, screenWidth - 20, 20}, "Scroll", TextFormat("%.0f", offsetX), &offsetX, 0.0f, maxOffsetX);

    // Draw the toggle button
    if (GuiButton((Rectangle){screenWidth / 2 - 50, screenHeight - 40, 100, 30}, downsamplingEnabled ? "Downsample: ON" : "Downsample: OFF"))
    {
        downsamplingEnabled = !downsamplingEnabled; // Toggle state on button press

        if (downsamplingEnabled)
        {
            samplingStep = 64;
        }
        else
        {
            samplingStep = 1;
        }
    }

    Rectangle panelBounds = {20, 20, 200, 200}; // Scroll panel bounds
    Rectangle content = {0, 0, 400, 400 + i};       // Content area
    Vector2 scroll = {0, 0};                    // Initial scroll position
    Rectangle view = {0, 0, 200, 200};          // View area (updated by GuiScrollPanel)

    // Chamada correta com 5 argumentos
    GuiScrollPanel(panelBounds, "", content, &scroll, &view);
}

// Main function
int main()
{

    // Using std::jthread in C++20 (automatically joins on destruction)
    std::jthread jt(threadCaptureAudioData);

    // Initialize raylib
    InitWindow(screenWidth, screenHeight, "Audio Plot with Downsampling Toggle");
    SetTargetFPS(60);

    // Main loop
    while (!WindowShouldClose())
    {
        // Clear the screen
        BeginDrawing();
        ClearBackground(BLACK);

        drawUI();

        plotSignal(screenWidth, screenHeight, bufferSize, samplingStep);

        EndDrawing();
    }

    // Clean up
    CloseWindow();

    jt.request_stop();

    return 0;
}

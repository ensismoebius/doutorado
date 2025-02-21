#include <iostream>
#include <vector>
#include <lsl_cpp.h>
#include <portaudio.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <GLFW/glfw3.h>

// Constants
const int SAMPLE_RATE = 44100;      // 44.1 kHz sample rate
const int FRAMES_PER_BUFFER = 512;  // Number of frames per buffer
const int NUM_CHANNELS = 1;         // Mono audio
const int PLOT_HISTORY_SIZE = 2000; // Number of samples to display in the plot

// Global variables
std::vector<float> audioData;              // Buffer to store audio data
lsl::stream_outlet *audioOutlet = nullptr; // LSL stream outlet

// Callback function for PortAudio (modified for LSL)
static int audioCallback(const void *inputBuffer, void *outputBuffer,
                         unsigned long framesPerBuffer,
                         const PaStreamCallbackTimeInfo *timeInfo,
                         PaStreamCallbackFlags statusFlags,
                         void *userData)
{
    // Cast input buffer to float
    const float *in = static_cast<const float *>(inputBuffer);

    // Append new audio data to the buffer
    for (int i = 0; i < framesPerBuffer; ++i)
    {
        audioData.push_back(in[i]);

        // Keep the buffer size limited to PLOT_HISTORY_SIZE
        if (audioData.size() > PLOT_HISTORY_SIZE)
        {
            audioData.erase(audioData.begin());
        }
    }

    // Push audio data to LSL
    if (audioOutlet)
    {
        audioOutlet->push_sample(&in[0]);
    }

    return paContinue;
}

// Initialize PortAudio
bool initPortAudio()
{
    PaError err = Pa_Initialize();
    if (err != paNoError)
    {
        std::cerr << "PortAudio initialization failed: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    // Open the default input stream
    PaStream *stream;
    err = Pa_OpenDefaultStream(&stream, NUM_CHANNELS, 0, paFloat32, SAMPLE_RATE, FRAMES_PER_BUFFER, audioCallback, nullptr);
    if (err != paNoError)
    {
        std::cerr << "PortAudio stream opening failed: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return false;
    }

    // Start the stream
    err = Pa_StartStream(stream);
    if (err != paNoError)
    {
        std::cerr << "PortAudio stream start failed: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        Pa_Terminate();
        return false;
    }

    return true;
}

// Clean up PortAudio
void cleanupPortAudio()
{
    Pa_Terminate();
}

// Main function
int main()
{
    // Initialize GLFW
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a window
    GLFWwindow *window = glfwCreateWindow(1280, 720, "Real-Time Audio Waveform with LSL", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Initialize LSL
    try
    {
        lsl::stream_info info("AudioStream", "Audio", NUM_CHANNELS, SAMPLE_RATE, lsl::cf_float32, "audio123");
        audioOutlet = new lsl::stream_outlet(info);
    }
    catch (std::exception &e)
    {
        std::cerr << "LSL initialization failed: " << e.what() << std::endl;
        return -1;
    }

    // Initialize PortAudio
    if (!initPortAudio())
    {
        return -1;
    }

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll events
        glfwPollEvents();

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create a window for the waveform
        ImGui::Begin("Audio Waveform");
        if (ImPlot::BeginPlot("Waveform", ImVec2(-1, -1)))
        {
            // Plot the audio data
            ImPlot::PlotLine("Audio", audioData.data(), audioData.size());

            // Set axis limits
            ImPlot::SetupAxes("Time", "Amplitude");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, PLOT_HISTORY_SIZE, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0f, 1.0f, ImGuiCond_Always);

            ImPlot::EndPlot();
        }
        ImGui::End();

        // Render ImGui
        ImGui::Render();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    cleanupPortAudio();
    if (audioOutlet)
    {
        delete audioOutlet;
    }
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
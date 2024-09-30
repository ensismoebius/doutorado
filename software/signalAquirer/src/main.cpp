#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h> // For GLFW and OpenGL

#include <raylib.h>
#include <alsa/asoundlib.h>

#include <vector>
#include <iostream>
#include <thread>
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

// Main function
int main()
{

    // Initialize GLFW
    if (!glfwInit())
        return -1;

    // Create a windowed mode window and its OpenGL context
    GLFWwindow *window = glfwCreateWindow(screenWidth, screenHeight, "ImGui Minimal Example", NULL, NULL);
    if (window == NULL)
        return -1;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize ImGui
    // IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    std::jthread jt(threadCaptureAudioData);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events
        glfwPollEvents();

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();

        // Create a simple window with a button
        ImGui::Begin("Hello, ImGui!");

        if (ImGui::Button("Press me"))
        {
            std::cout << "Teste" << std::endl;
            teste = !teste;
        }

        if(teste){
            ImGui::Text("Teste");
            ImGui::Text("Teste");
            ImGui::Text("Teste");
            ImGui::Text("Teste");
            ImGui::Text("Teste");
        }

        ImGui::End();

        // Rendering
        ImGui::Render();
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    jt.request_stop();
    return 0;
}

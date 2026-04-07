/**
 * @file imgui_glfw.cpp
 * @brief ImGui+GLFW application wrapper (snake_case translation unit).
 */

#include "nn/utility/imgui_glfw.hpp"

#include "nn/logging/Logger.hpp"

ImGuiApp::ImGuiApp(const std::string& title, int width, int height)
    : window(nullptr), title(title), width(width), height(height)
{
}

ImGuiApp::~ImGuiApp()
{
    shutdown();
}

void ImGuiApp::glfw_error_callback(int error, const char* description)
{
    NN_LOG_ERROR("GLFW Error " + std::to_string(error) + ": " + std::string(description));
}

bool ImGuiApp::initialize_glfw()
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
    {
        NN_LOG_ERROR("Failed to initialize GLFW");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window)
    {
        NN_LOG_ERROR("Failed to create GLFW window");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable V-Sync
    return true;
}

bool ImGuiApp::initialize_imgui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    const ImGuiIO& io = ImGui::GetIO();
    (void) io;

    if (!ImGui_ImplGlfw_InitForOpenGL(window, true) || !ImGui_ImplOpenGL3_Init(glsl_version))
    {
        NN_LOG_ERROR("Failed to initialize ImGui");
        return false;
    }

    return true;
}

void ImGuiApp::prepare_frame()
{
    glfwPollEvents();

    int frameWidth, frameHeight;
    glfwGetFramebufferSize(window, &frameWidth, &frameHeight);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(frameWidth, frameHeight));

    ImGui::Begin("Fullscreen Window",
        nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse);
}

void ImGuiApp::render_frame()
{
    ImGui::End();

    ImGui::Render();
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
}

void ImGuiApp::shutdown()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}

bool ImGuiApp::initialize()
{
    return initialize_glfw() && initialize_imgui();
}

void ImGuiApp::run(const std::function<void()>& uiCode)
{
    while (!glfwWindowShouldClose(window))
    {
        prepare_frame();

        // Execute the provided UI code
        uiCode();

        render_frame();
    }
}
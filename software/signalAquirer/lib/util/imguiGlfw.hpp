#ifndef IMGUI_GLFW_H
#define IMGUI_GLFW_H

#include <iostream>
#include <functional>

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

class ImGuiApp
{
public:
    ImGuiApp(const std::string &title, int width, int height);
    ~ImGuiApp();
    bool initialize();
    void run(const std::function<void()> &uiCode);

private:
    GLFWwindow *window;
    std::string title;
    int width;
    int height;
    const char *glsl_version = "#version 330";

    static void glfw_error_callback(int error, const char *description);
    bool initializeGLFW();
    bool initializeImGui();
    void prepareFrame();
    void renderFrame();
    void shutdown();
};

#endif
#ifndef IMGUI_GLFW_H
#define IMGUI_GLFW_H

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <functional>
#include <iostream>

class ImGuiApp
{
   public:
    ImGuiApp(const std::string& title, int width, int height);
    ~ImGuiApp();
    auto initialize() -> bool;
    void run(const std::function<void()>& uiCode);

   private:
    GLFWwindow* window;
    std::string title;
    int width;
    int height;
    const char* glsl_version = "#version 330";

    static void glfw_error_callback(int error, const char* description);
    auto initializeGLFW() -> bool;
    auto initializeImGui() -> bool;
    void prepareFrame();
    void renderFrame();
    void shutdown();
};

#endif
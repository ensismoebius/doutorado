#ifndef NN_UTILITY_IMGUI_GLFW_HPP
#define NN_UTILITY_IMGUI_GLFW_HPP

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <functional>
#include <iostream>

/**
 * @file imgui_glfw.hpp
 * @brief Minimal RAII wrapper for an ImGui + GLFW + OpenGL3 application loop.
 *
 * This is used by visualization demos. It hides the boilerplate required to:
 * - initialize GLFW and OpenGL context
 * - set up Dear ImGui backends
 * - run a render loop executing user-provided UI code
 */

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
    auto initialize_glfw() -> bool;
    auto initialize_imgui() -> bool;
    void prepare_frame();
    void render_frame();
    void shutdown();
};

#endif
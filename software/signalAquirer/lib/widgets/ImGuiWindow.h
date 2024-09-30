#ifndef IMGUIWINDOW_H
#define IMGUIWINDOW_H

#pragma once

#include <string>
#include <cstdlib>
#include <imgui.h>
#include <implot.h>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

template <typename DeriviedClass>
class ImGuiWindow
{
public:
    ImGuiWindow(const std::string &title, ImVec2 dimensions, ImVec4 color);
    ~ImGuiWindow();
    void update();

private:
    GLFWwindow *window;
};

#endif
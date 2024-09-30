#ifndef IMGUIWINDOW_H
#define IMGUIWINDOW_H

#pragma once

#include <string>
#include <cstdlib>
#include <imgui.h>
#include <implot.h>
#include <stdexcept>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

template <typename DerivedClass>
class CustomImGuiWindow
{
public:
    CustomImGuiWindow(const std::string &title, ImVec2 dimensions, ImVec4 color);
    ~CustomImGuiWindow();
    void startUp();
    void update();
    void run();

private:
    GLFWwindow *window;
    ImVec4 clear_color = ImVec4(0.1058, 0.1137f, 0.1255f, 1.00f);
};

#endif
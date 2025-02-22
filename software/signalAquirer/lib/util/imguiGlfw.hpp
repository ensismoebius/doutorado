#ifndef IMGUI_GLFW
#define IMGUI_GLFW

#include <iostream>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h> // Inclui os headers do OpenGL

void glfw_error_callback(int error, const char *description);
bool InitializeGLFW();
bool InitializeImGui();
bool windowShouldClose();
void prepareNextFrame();
void renderNextFrame();
void terminateWindow();
bool initializeWindow();

#endif
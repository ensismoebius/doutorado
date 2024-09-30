#include "ImGuiWindow.h"

template <typename DerivedClass>
ImGuiWindow<DerivedClass>::ImGuiWindow(const std::string &title, ImVec2 dimensions, ImVec4 color)
{
    // Initialize GLFW
    if (!glfwInit())
        std::exit(-1);

    // Create a windowed mode window and its OpenGL context
    this->window = glfwCreateWindow(
        static_cast<int>(dimensions.x),
        static_cast<int>(dimensions.y),
        title.c_str(),
        NULL,
        NULL);

    if (window == NULL)
        std::exit(-1);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
}

template <typename DerivedClass>
ImGuiWindow<DerivedClass>::~ImGuiWindow()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}

template <typename DerivedClass>
void ImGuiWindow<DerivedClass>::update()
{
    static_cast<const DerivedClass *>(this)->update();
}


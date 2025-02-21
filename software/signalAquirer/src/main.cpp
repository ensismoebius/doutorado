#include <iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h> // Inclui os headers do OpenGL

const char *glsl_version = "#version 330";
GLFWwindow *window;

void glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

bool InitializeGLFW()
{
    // Inicializa GLFW
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
    {
        return false;
    }

    // Configurações do OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Para compatibilidade com MacOS
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    // Criação da janela
    window = glfwCreateWindow(800, 600, "ImGui Fullscreen Frame", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Ativa V-Sync

    return true;
}

bool InitializeImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext(); // Criando contexto do ImGui

    ImGuiIO &io = ImGui::GetIO();
    (void)io;

    // Configuração do backend para GLFW e OpenGL
    if (!ImGui_ImplGlfw_InitForOpenGL(window, true) || !ImGui_ImplOpenGL3_Init(glsl_version))
    {
        return false;
    }

    return true;
}

int main()
{
    if (!InitializeGLFW())
    {
        return -1;
    }
    if (!InitializeImGui())
    {
        glfwTerminate();
        return -1;
    }

    // Loop principal
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Obtém o tamanho da janela
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        // Começa um novo frame do ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Janela fullscreen do ImGui
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(width, height));

        ImGui::Begin("Fullscreen Window", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        ImGui::Text("Esta janela do ImGui ocupa toda a janela da aplicação.");
        ImGui::Text("Redimensione a janela e veja como a UI se adapta.");
        ImGui::Separator();
        ImGui::Text("Adicione mais elementos de UI aqui...");

        ImGui::End();

        // Renderização
        ImGui::Render();
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

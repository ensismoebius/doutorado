#include "imguiGlfw.hpp"

const char *glsl_version = "#version 330";
GLFWwindow *window;

bool windowShouldClose()
{
    return glfwWindowShouldClose(window);
}

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

void prepareNextFrame()
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
}

void renderNextFrame()
{
    ImGui::End();

    // Renderização
    ImGui::Render();
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
}

void terminateWindow()
{
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}

bool initializeWindow()
{
    if (!InitializeGLFW())
    {
        return false;
    }
    if (!InitializeImGui())
    {
        glfwTerminate();
        return false;
    }

    return true;
}

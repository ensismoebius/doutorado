#include <cmath>
#include <string>
#include <vector>
#include <imgui.h>
#include <implot.h>
#include "../lib/widgets/CustomImGuiWindow.cpp"

class Window : public CustomImGuiWindow<Window> // Explicit public inheritance
{
private:
    std::string title;
    ImVec2 dimensions;
    ImVec4 color;

public:
    float my_color[4] = {255, 255, 255, 255};
    float *data;
    float meu_valor = 0.0f;

public:
    // Constructor initializes the base class with the required parameters
    Window(const std::string &title, ImVec2 dimensions, ImVec4 color)
        : CustomImGuiWindow<Window>(title, dimensions, color),
          title(title),
          dimensions(dimensions),
          color(color)
    {
    }

    ~Window() = default; // Explicitly use default destructor

    void startUp()
    {
        // Initialize resources or variables
    }

    void update()
    {
        // Create a window called "My First Tool", with a menu bar.
        ImGui::Begin("My First Tool", nullptr, ImGuiWindowFlags_MenuBar);
        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Open..", "Ctrl+O"))
                { /* Do stuff */
                }
                if (ImGui::MenuItem("Save", "Ctrl+S"))
                { /* Do stuff */
                }
                if (ImGui::MenuItem("Close", "Ctrl+W"))
                {
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        // Edit a color stored as 4 floats
        ImGui::ColorEdit4("Color", my_color);

        // Altere a cor do gráfico
        // ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(my_color[0], my_color[1], my_color[2], my_color[3]));
        // ImGui::PlotLines("Samples", data, 4096, 0, NULL, -meu_valor, meu_valor, ImVec2(0, meu_valor), 1);
        // ImGui::PopStyleColor();

        // Plotagem de gráfico
        ImPlot::BeginPlot("Gráfico de Linha");
        static float x_values[] = {0, 1, 2, 3, 4};
        static float y_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        // PlotLine(const char* label_id, const T* values, int count, double xscale=1, double xstart=0, ImPlotLineFlags flags=0, int offset=0, int stride=sizeof(T));


        ImPlot::PlotLine("Valores", data, 64);
        ImPlot::EndPlot();

        ImGui::SliderFloat("Controle de Valor", &meu_valor, 0.0f, 1024.0f); // Cria um slider de 0.0 até 1.0

        // Display contents in a scrolling region
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Important Stuff");
        ImGui::BeginChild("Scrolling");
        for (int n = 0; n < 50; n++)
            ImGui::Text("%04d: Some text", n);
        ImGui::EndChild();
        ImGui::End();
    }
};

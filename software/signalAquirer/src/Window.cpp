#include "../lib/widgets/CustomImGuiWindow.cpp"
#include <string>
#include <imgui.h>

class Window : public CustomImGuiWindow<Window> // Explicit public inheritance
{
private:
    std::string title;
    ImVec2 dimensions;
    ImVec4 color;

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
        ImGui::Begin(title.c_str()); // Create a new ImGui window

        // Set window size and background color
        ImGui::SetWindowSize(dimensions);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, color);

        // Your ImGui code here (e.g., UI elements)
        ImGui::Text("Hello, this is your window!");

        ImGui::PopStyleColor(); // Reset color to default
        ImGui::End();           // End the ImGui window
    }
};

#include <iostream>
#include "../lib/util/imguiGlfw.hpp"

int main()
{
    ImGuiApp app("ImGui Fullscreen Frame", 800, 600);
    if (!app.initialize())
    {
        return -1;
    }
    // Pass a lambda to run
    app.run([]()
            {
            ImGui::Text("Hello, ImGui!");
            if (ImGui::Button("Click Me")) {
                std::cout << "Button clicked!" << std::endl;
            } });

    return 0;
}

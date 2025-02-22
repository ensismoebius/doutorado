#include <iostream>
#include "../lib/util/imguiGlfw.hpp"

int main()
{
    if (!initializeWindow())
        return -1;

    // Loop principal
    while (!windowShouldClose())
    {
        prepareNextFrame();

        ImGui::Text("Adicione mais elementos de UI aqui...");

        renderNextFrame();
    }

    terminateWindow();
    return 0;
}

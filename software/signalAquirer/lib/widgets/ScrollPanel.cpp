#include "ScrollPanel.h"
#include <iostream>
#include <algorithm>

ScrollPanel::ScrollPanel(const std::string &title, ImVec2 dimensions, ImVec4 color)
    : title(title), dimensions(dimensions), color(color), scroll(0.0f, 0.0f), contentSize(400, 200), zoomLevel(1.0f), scrollSpeed(1.0f)
{
    // Initialize the scroll bars dimensions
    horizontalBar = ImRect(20, dimensions.y + dimensions.y + 20, dimensions.x - 20, dimensions.y + dimensions.y + 40);
    verticalBar = ImRect(0, dimensions.y + 20, 20, dimensions.y - 20);
}

void ScrollPanel::handleInput()
{
    // Check if the panel is focused (mouse over it)
    ImVec2 mousePos = ImGui::GetMousePos();
    bool isInFocus = ImGui::IsMouseHoveringRect(dimensions,
                                                ImVec2(dimensions.x + dimensions.x, dimensions.y + dimensions.y), false);

    // Allow zoom only if the panel is in focus
    if (isInFocus)
    {
        ImVec2 panelCenter = {dimensions.x / 2.0f, dimensions.y / 2.0f};
        ImVec2 preZoomCenter = {(panelCenter.x - scroll.x) / zoomLevel, (panelCenter.y - scroll.y) / zoomLevel};

        // Handle mouse wheel zoom
        float mouseWheelMove = ImGui::GetIO().MouseWheel;
        if (mouseWheelMove != 0)
        {
            zoomLevel += mouseWheelMove * 0.1f;    // Adjust sensitivity
            zoomLevel = std::max(0.1f, zoomLevel); // Avoid negative or too small zoom
            scroll.x = panelCenter.x - preZoomCenter.x * zoomLevel;
            scroll.y = panelCenter.y - preZoomCenter.y * zoomLevel;
        }

        // Handle keyboard zoom
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow))
        {
            zoomLevel += 0.1f;
            scroll.x = panelCenter.x - preZoomCenter.x * zoomLevel;
            scroll.y = panelCenter.y - preZoomCenter.y * zoomLevel;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow))
        {
            zoomLevel = std::max(0.1f, zoomLevel - 0.1f);
            scroll.x = panelCenter.x - preZoomCenter.x * zoomLevel;
            scroll.y = panelCenter.y - preZoomCenter.y * zoomLevel;
        }
    }
}

bool ScrollPanel::draw(std::vector<short> *data)
{
    return true; // Change based on your logic for returning if scrolling is active
}

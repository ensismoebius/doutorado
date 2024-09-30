#ifndef __SCROLLPANEL_H__
#define __SCROLLPANEL_H__

#include <imgui.h>
#include <imgui_internal.h>

#include <vector>
#include <string>

class ScrollPanel
{
private:
    std::string title;          // Title of the scroll panel
    ImVec2 dimensions;          // Dimensions of the panel
    ImVec2 scroll;              // Scrolling position
    ImVec2 contentSize;         // Size of the content
    ImVec4 color;               // Color for the plot lines (RGBA)

    ImRect horizontalBar;       // Rectangle for the horizontal scroll bar
    ImRect verticalBar;         // Rectangle for the vertical scroll bar

public:
    bool autoScroll = false;    // Enable automatic scrolling
    bool isScrolling = false;    // Is the user scrolling?
    float scrollSpeed = 1.0f;    // Scroll speed
    float zoomLevel = 1.0f;      // Initial zoom level

public:
    // Constructor
    ScrollPanel(const std::string& title, ImVec2 dimensions, ImVec4 color);

    // Method to handle user input for scrolling and zooming
    void handleInput();

    // Method to draw the scroll panel and its contents
    bool draw(std::vector<short>* data);
};

#endif // __SCROLLPANEL_H__

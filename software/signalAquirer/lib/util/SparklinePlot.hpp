#ifndef SPARKLINE_PLOT_HPP
#define SPARKLINE_PLOT_HPP

#include <vector>
#include <algorithm>

#include "imgui.h"
#include "../lib/implot/implot.h"

class SparklinePlot
{
public:
    // Renders the sparkline plot with the given ID and sample data
    static void Render(const char *id, const std::vector<float> &samples, unsigned int sampleRate, bool showDetails = false);

private:
    static inline float scrollOffset = 0.0f;    // Horizontal scroll offset
    static inline float previousMouseX = -1.0f; // Previous X position of the mouse (for dragging)
    static inline double selectionStart = 1.0f; // Start position of the selection range
    static inline double selectionEnd = 6.0f;   // End position of the selection range
};

#endif // SPARKLINE_PLOT_HPP

void SparklinePlot::Render(const char *id, const std::vector<float> &samples, unsigned int sampleRate, bool showDetails)
{
    static float timeStep = 0;
    if (sampleRate != 0)
    {
        timeStep = 1.0f / sampleRate;
    }
    else
    {
        timeStep = 0;
    }

    float totalDuration = samples.size() * timeStep;
    float displayDuration = 6.0f;
    double xMin = std::max(0.0f, totalDuration - displayDuration - scrollOffset);
    double xMax = xMin + displayDuration;

    struct DataWrapper
    {
        const std::vector<float> &samples;
        float timeStep;
    };

    auto getter = [](int idx, void *data) -> ImPlotPoint
    {
        DataWrapper *wrapper = static_cast<DataWrapper *>(data);
        return ImPlotPoint(idx * wrapper->timeStep, wrapper->samples[idx]);
    };

    DataWrapper wrapper = {samples, timeStep};

    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    if (ImPlot::BeginPlot(id, ImVec2(-1, 100), ImPlotFlags_CanvasOnly))
    {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxisLimits(ImAxis_X1, xMin, xMax, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.2f, 1.2f);
        ImPlot::SetNextLineStyle(ImVec4(0, 1, 0, .5), 1.5f);

        ImPlot::PlotLineG("Signal", getter, &wrapper, samples.size());

        // Handle selection dragging with minimal redundant operations
        if (ImPlot::DragLineX(0, &selectionStart, ImVec4(1, 0, 0, 1), 1.5f))
        {
            selectionStart = std::clamp(selectionStart, xMin, xMax);
            selectionStart = std::min(selectionStart, selectionEnd - 0.001f);
        }
        if (ImPlot::DragLineX(1, &selectionEnd, ImVec4(0, 0, 1, 1), 1.5f))
        {
            selectionEnd = std::clamp(selectionEnd, xMin, xMax);
            selectionEnd = std::max(selectionEnd, selectionStart + 0.001f);
        }

        // Optimize selection line rendering by computing pixel positions once
        ImVec2 p1 = ImPlot::PlotToPixels(ImVec2(selectionStart, 0.0f));
        ImVec2 p2 = ImPlot::PlotToPixels(ImVec2(selectionEnd, 0.0f));
        ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 0, 255, 20), ImPlot::GetPlotSize().y);

        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();

    // Optimize scrolling logic by resetting previousMouseX when mouse button is released
    ImVec2 mousePos = ImGui::GetMousePos();
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right))
    {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);

        if (previousMouseX != -1.0f)
        {
            float deltaX = mousePos.x - previousMouseX;
            scrollOffset += deltaX / ImGui::GetIO().DisplaySize.x * totalDuration;
            scrollOffset = std::clamp(scrollOffset, 0.0f, totalDuration - displayDuration);
        }
        previousMouseX = mousePos.x;
    }
    else
    {
        previousMouseX = -1.0f; // Reset when mouse button is released
    }

    float maxScrollOffset = totalDuration - displayDuration;
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
    if (showDetails)
    {
        ImGui::Text("Selection Start: %.2f sec", selectionStart);
        ImGui::Text("Selection End: %.2f sec", selectionEnd);
        ImGui::SliderFloat("##Scroll", &scrollOffset, maxScrollOffset, 0.0f, "", ImGuiSliderFlags_None);
    }
    ImGui::PopItemWidth();
}

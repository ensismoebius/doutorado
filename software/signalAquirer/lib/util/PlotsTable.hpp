#ifndef PLOTS_TABLE_H
#define PLOTS_TABLE_H

#include <vector>
#include <memory>
#include <algorithm>

#include "../lib/implot/implot.h"

using namespace std;

class PlotsTable
{
private:
    const static ImGuiTableFlags flags = ImGuiTableFlags_BordersOuter |
                                         ImGuiTableFlags_BordersV |
                                         ImGuiTableFlags_RowBg |
                                         ImGuiTableFlags_Resizable |
                                         ImGuiTableFlags_Reorderable;

public:
    PlotsTable();
    ~PlotsTable();

    // Correção no parâmetro: Recebe um vetor de shared_ptr
    void plotAll(const vector<vector<float>> &table, const unsigned int TIMELINE_SIZE, const unsigned int SAMPLE_RATE);

    static void Sparkline(
        const char *id,
        const std::vector<float> &samples,
        const unsigned int TIMELINE_SIZE,
        const unsigned int SAMPLE_RATE);
};

#endif

PlotsTable::PlotsTable() {}

PlotsTable::~PlotsTable() {}

void PlotsTable::plotAll(
    const vector<vector<float>> &table,
    const unsigned int TIMELINE_SIZE,
    const unsigned int SAMPLE_RATE)
{
    if (ImGui::BeginTable("##table", 2, flags, ImVec2(-1, 0)))
    {
        ImGui::TableSetupColumn("Electrode", ImGuiTableColumnFlags_WidthFixed, 75.0f);
        ImGui::TableSetupColumn("EMG Signal");

        ImGui::TableHeadersRow();
        ImPlot::PushColormap(ImPlotColormap_Cool);

        for (int rowIndex = 0; rowIndex < table.size(); ++rowIndex)
        {
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("EMG %d", rowIndex);

            ImGui::TableSetColumnIndex(1);
            ImGui::PushID(rowIndex);

            Sparkline(
                "##spark",
                table[rowIndex],
                TIMELINE_SIZE,
                SAMPLE_RATE);

            ImGui::PopID();
        }

        ImPlot::PopColormap();
        ImGui::EndTable();
    }
}

void PlotsTable::Sparkline(
    const char *id,
    const std::vector<float> &samples,
    const unsigned int TIMELINE_SIZE,
    const unsigned int SAMPLE_RATE)
{
    static const float timeStep = 1.0f / SAMPLE_RATE;
    static float scrollOffset = 0.0f; // Track user scroll position

    float totalDuration = samples.size() * timeStep; // Total duration of data
    float displayDuration = 6.0f;                    // Duration of data displayed in the plot (6 seconds)

    // Set x-axis range for the plot to show last 6 seconds
    double xMin = max(0.0f, totalDuration - displayDuration - scrollOffset);
    double xMax = xMin + displayDuration;

    static double selectionStart = xMin + 1.0f; // Posição inicial da seleção
    static double selectionEnd = xMin + 2.0f;   // Posição final da seleção

    static auto getter = [](int idx, void *data) -> ImPlotPoint
    {
        float *samples = static_cast<float *>(data);
        return ImPlotPoint(idx * timeStep, samples[idx]);
    };

    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    if (ImPlot::BeginPlot(id, ImVec2(-1, 100), ImPlotFlags_CanvasOnly))
    {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxisLimits(ImAxis_X1, xMin, xMax, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.2f, 1.2f);
        ImPlot::SetNextLineStyle(ImVec4(0, 1, 0, .5), 1.5f);

        ImPlot::PlotLineG(
            "Audio Signal",
            getter,
            (void *)samples.data(),
            samples.size());

        // Adiciona linhas móveis para selecionar início e fim do intervalo
        if (ImPlot::DragLineX(0, &selectionStart, ImVec4(1, 0, 0, 1), 1.5f))
        {
            selectionStart = std::clamp(selectionStart, xMin, xMax);
            // Ensure selectionStart is smaller than selectionEnd and not equal
            if (selectionStart >= selectionEnd)
            {
                selectionStart = selectionEnd - 0.001f; // Ensure a slight difference
            }
        }

        if (ImPlot::DragLineX(1, &selectionEnd, ImVec4(0, 0, 1, 1), 1.5f))
        {
            selectionEnd = std::clamp(selectionEnd, xMin, xMax);
            // Ensure selectionEnd is greater than selectionStart and not equal
            if (selectionEnd <= selectionStart)
            {
                selectionEnd = selectionStart + 0.001f; // Ensure a slight difference
            }
        }

        ////////////////////// Draws the rectangle
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 0.0f, 1.0f, 0.2f)); // Set shading color to semi-transparent blue

        // Define the rectangle's position and size in plot coordinates
        ImVec2 p1 = ImPlot::PlotToPixels(ImVec2(selectionStart, 0.0f));
        ImVec2 p2 = ImPlot::PlotToPixels(ImVec2(selectionEnd, 0.0f));

        // Draw the rectangle (outline)
        ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 0, 255, 125), 0.0f);

        ImPlot::PopStyleColor(); // Reset fill color
        ////////////////////// Draws the rectangle - end

        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();

    // Exibe valores da seleção
    ImGui::Text("Selection Start: %.2f sec", selectionStart);
    ImGui::Text("Selection End: %.2f sec", selectionEnd);

    // Scroll control
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))
        scrollOffset = min(scrollOffset + 0.5f, totalDuration - displayDuration);
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow))
        scrollOffset = max(0.0f, scrollOffset - 0.5f);

    float maxScrollOffset = totalDuration - displayDuration; // Max scroll position
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::SliderFloat("##Scroll", &scrollOffset, maxScrollOffset, 0.0f, "%.2f sec", ImGuiSliderFlags_None);
    ImGui::PopItemWidth();
}

#ifndef PLOTS_TABLE_H
#define PLOTS_TABLE_H

#include <vector>
#include <memory>

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

    static auto getter = [](int idx, void *data) -> ImPlotPoint
    {
        float *samples = static_cast<float *>(data); // Directly cast to float array
        return ImPlotPoint(idx * timeStep, samples[idx]);
    };

    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    if (ImPlot::BeginPlot(id, ImVec2(-1, 100), ImPlotFlags_CanvasOnly))
    {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0f, TIMELINE_SIZE, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.2f, 1.2f);
        ImPlot::SetNextLineStyle(ImVec4(0, 1, 0, .5), 1.5f); // Green line, 1.5px thick

        ImPlot::PlotLineG(
            "Audio Signal",
            getter,
            (void *)samples.data(),
            static_cast<int>(samples.size()));

        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
}

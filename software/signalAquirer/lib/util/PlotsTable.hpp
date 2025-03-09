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

    static float data[100];

    vector<shared_ptr<vector<float>>> table;

public:
    PlotsTable();
    ~PlotsTable();
    void plotAll();
    void Sparkline(const char *id, const shared_ptr<vector<float>> values, int count, float min_v, float max_v, int offset, const ImVec4 &col, const ImVec2 &size);
};

#endif

PlotsTable::PlotsTable()
{
}

PlotsTable::~PlotsTable()
{
}

void PlotsTable::plotAll()
{

    if (ImGui::BeginTable("##table", 3, flags, ImVec2(-1, 0)))
    {
        ImGui::TableSetupColumn("Electrode", ImGuiTableColumnFlags_WidthFixed, 75.0f);
        ImGui::TableSetupColumn("Voltage", ImGuiTableColumnFlags_WidthFixed, 75.0f);
        ImGui::TableSetupColumn("EMG Signal");

        ImGui::TableHeadersRow();

        ImPlot::PushColormap(ImPlotColormap_Cool);

        for (int rowIndex = 0; rowIndex < this->table.size(); ++rowIndex)
        {
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("EMG %d", rowIndex);

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.3f V", this->table.at(rowIndex)->at(0));

            ImGui::TableSetColumnIndex(2);
            ImGui::PushID(rowIndex);

            Sparkline(
                "##spark",
                this->table.at(rowIndex),
                100,
                -1.0f,
                1.0f,
                100,
                ImPlot::GetColormapColor(rowIndex),
                ImVec2(-1, 100));

            ImGui::PopID();
        }

        ImPlot::PopColormap();
        ImGui::EndTable();
    }
}

void PlotsTable::Sparkline(const char *id, const shared_ptr<vector<float>> values, int count, float min_v, float max_v, int offset, const ImVec4 &col, const ImVec2 &size)
{
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    if (ImPlot::BeginPlot(id, size, ImPlotFlags_CanvasOnly))
    {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxesLimits(0, count - 1, min_v, max_v, ImGuiCond_Always);
        ImPlot::SetNextLineStyle(col);
        ImPlot::SetNextFillStyle(col, 0.25);
        ImPlot::PlotLine(id, values, count, 1, 0, ImPlotLineFlags_Shaded, offset);
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
}
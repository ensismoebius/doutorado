#ifndef PLOTS_TABLE_H
#define PLOTS_TABLE_H

#include <vector>
#include <map>
#include <memory>
#include <algorithm>

#include "../lib/implot/implot.h"
#include "../lib/util/SparklinePlot.hpp"

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
        ImGui::TableSetupColumn("Origin", ImGuiTableColumnFlags_WidthFixed, 75.0f);
        ImGui::TableSetupColumn("Signal");

        ImGui::TableHeadersRow();
        ImPlot::PushColormap(ImPlotColormap_Cool);

        for (int rowIndex = 0; rowIndex < table.size(); ++rowIndex)
        {
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("EMG %d", rowIndex);

            ImGui::TableSetColumnIndex(1);
            ImGui::PushID(rowIndex);

            SparklinePlot::Render("##spark", table[rowIndex], SAMPLE_RATE);

            ImGui::PopID();
        }

        ImPlot::PopColormap();
        ImGui::EndTable();
    }
}

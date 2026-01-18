#ifndef PLOTS_TABLE_H
#define PLOTS_TABLE_H

#include <vector>
#include <map>
#include <memory>
#include <algorithm>

#include "../lib/implot/implot.h"
#include "../lib/util/SparklinePlot.hpp"

using namespace std;

typedef struct
{
    vector<float> signal;
    size_t sampleRate;
} Signal;

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
    void plotAll(const map<string, Signal> &table, const unsigned int TIMELINE_SIZE);
};

#endif

PlotsTable::PlotsTable() {}

PlotsTable::~PlotsTable() {}

void PlotsTable::plotAll(
    const map<string, Signal> &table,
    const unsigned int TIMELINE_SIZE)
{
    if (ImGui::BeginTable("##table", 2, flags, ImVec2(-1, 0)))
    {
        ImGui::TableSetupColumn("Origin", ImGuiTableColumnFlags_WidthFixed, 75.0f);
        ImGui::TableSetupColumn("Signal");

        ImGui::TableHeadersRow();
        ImPlot::PushColormap(ImPlotColormap_Cool);

        bool showDetails = true;

        for (auto it = table.begin(); it != table.end(); ++it)
        {
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", it->first.c_str()); // Access the key (e.g., "EMG 1")

            ImGui::TableSetColumnIndex(1);
            ImGui::PushID(it->first.c_str()); // Use key as the ID

            SparklinePlot::Render("##spark", it->second.signal, it->second.sampleRate, showDetails); // Access the value (the vector)
            showDetails = false;

            ImGui::PopID();
        }

        ImPlot::PopColormap();
        ImGui::EndTable();
    }
}

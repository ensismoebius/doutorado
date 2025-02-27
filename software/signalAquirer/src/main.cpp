#include <iostream>
#include <memory>
#include <vector>
#include <random>

#include "../lib/util/imguiGlfw.hpp"
#include "../lib/util/AudioCapture.hpp"
#include "../lib/implot/implot.h"

// Global state for audio capture
std::unique_ptr<AudioCapture> audio_capture;
bool is_capturing = false;
std::vector<float> audio_samples;
std::string error_message;

template <typename T>
inline T RandomRange(T min, T max)
{
    T scale = rand() / (T)RAND_MAX;
    return min + scale * (max - min);
}

void Sparkline(const char *id, const float *values, int count, float min_v, float max_v, int offset, const ImVec4 &col, const ImVec2 &size)
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

void Demo_Tables()
{
    static ImGuiTableFlags flags = ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
                                   ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable;
    static bool anim = true;
    static int offset = 0;
    ImGui::BulletText("Plots can be used inside of ImGui tables as another means of creating subplots.");
    ImGui::Checkbox("Animate", &anim);

    if (anim)
        offset = (offset + 1) % 100;

    if (ImGui::BeginTable("##table", 3, flags, ImVec2(-1, 0)))
    {
        ImGui::TableSetupColumn("Electrode", ImGuiTableColumnFlags_WidthFixed, 75.0f);
        ImGui::TableSetupColumn("Voltage", ImGuiTableColumnFlags_WidthFixed, 75.0f);
        ImGui::TableSetupColumn("EMG Signal");
        ImGui::TableHeadersRow();
        ImPlot::PushColormap(ImPlotColormap_Cool);
        for (int row = 0; row < 10; row++)
        {
            ImGui::TableNextRow();
            static float data[100];
            srand(row);
            for (int i = 0; i < 100; ++i)
                data[i] = RandomRange(0.0f, 10.0f);
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("EMG %d", row);
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.3f V", data[offset]);
            ImGui::TableSetColumnIndex(2);
            ImGui::PushID(row);
            Sparkline("##spark", data, 100, 0, 11.0f, offset, ImPlot::GetColormapColor(row), ImVec2(-1, 35));
            ImGui::PopID();
        }
        ImPlot::PopColormap();
        ImGui::EndTable();
    }
}

void audio_visualization()
{
    if (!audio_capture || !is_capturing || audio_samples.empty())
        return;

    // Create time axis data
    static std::vector<float> time;
    if (time.size() != audio_samples.size())
    {
        time.resize(audio_samples.size());
        const float dt = 1.0f / AudioCapture::SAMPLE_RATE;
        for (size_t i = 0; i < time.size(); ++i)
        {
            time[i] = static_cast<float>(i) * dt;
        }
    }

    // Configure plot
    const float plot_height = 400.0f;
    const ImVec2 plot_size(-1, plot_height);

    if (ImPlot::BeginPlot("Audio Waveform", "Time (s)", "Amplitude", plot_size))
    {
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0f, time.back(), ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -2.0f, 2.0f);
        ImPlot::SetNextLineStyle(ImVec4(0, 1, 0, 1), 1.5f); // Green line, 1.5px thick

        ImPlot::PlotLine("Audio Signal",
                         time.data(),
                         audio_samples.data(),
                         static_cast<int>(audio_samples.size()));

        ImPlot::EndPlot();
    }
}

void widgets()
{
    if (ImGui::Button(is_capturing ? "Stop Capture" : "Start Capture"))
    {
        if (!is_capturing)
        {
            audio_capture = std::make_unique<AudioCapture>();
            if (!audio_capture->start())
            {
                error_message = audio_capture->last_error();
                audio_capture.reset();
            }
            else
            {
                is_capturing = true;
                audio_samples.clear();
            }
        }
        else
        {
            audio_capture->stop();
            is_capturing = false;
            audio_capture.reset();
        }
    }

    if (ImGui::Button("Clear Display"))
    {
        audio_samples.clear();
    }

    if (!error_message.empty())
    {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Error: %s", error_message.c_str());
        if (ImGui::Button("Clear Error"))
        {
            error_message.clear();
        }
    }

    // Update audio buffer
    if (is_capturing && audio_capture)
    {
        auto new_samples = audio_capture->get_available_samples();
        if (!new_samples.empty())
        {
            audio_samples.insert(audio_samples.end(), new_samples.begin(), new_samples.end());

            // Keep last 5 seconds of data
            const size_t max_samples = 5 * AudioCapture::SAMPLE_RATE;
            if (audio_samples.size() > max_samples)
            {
                audio_samples.erase(audio_samples.begin(),
                                    audio_samples.end() - max_samples);
            }
        }
    }

    // Show visualization
    audio_visualization();

    Demo_Tables();
}

int main()
{
    ImGuiApp app("Audio Visualizer", 1280, 720);
    if (!app.initialize())
        return -1;

    ImPlot::CreateContext();
    app.run(widgets);
    ImPlot::DestroyContext();

    return 0;
}
#include <iostream>
#include <memory>
#include <vector>
#include <random>

#include "../lib/util/CapturerManager.hpp"
#include "../lib/util/AudioCapture.hpp"
#include "../lib/util/MockCapturer.hpp"
#include "../lib/util/SignalPlotter.hpp"

#include "../lib/util/imguiGlfw.hpp"
#include "../lib/implot/implot.h"
#include "main.h"

#define TIMELINE_SIZE 10

using namespace std;

// Signal capture manager and signal capturer
shared_ptr<ICapturer> audioCapturer = make_shared<AudioCapture>();
CapturerManager capturerManager;

// Global state for audio capture
std::vector<float> audioSamples;
std::string errorMessage;

inline void toggleAudioCapture(
    CapturerManager *captureManager,
    std::vector<float> samples,
    std::string &errorMessages)
{
    // If capturing the audio then stop it
    if (captureManager->isCapturing())
    {
        captureManager->stopCapturing();
        return;
    }

    // If NOT capturing then starts the audio capturing
    if (captureManager->startCapturing())
    {
        samples.clear();
        return;
    }

    // If NOT capturing and can't start capturing,
    // stores an error message
    errorMessages = captureManager->getErrors();
}

void getSamples(vector<float> &audioSamples)
{
    // Update audio buffer
    if (capturerManager.isCapturing())
    {
        const auto new_samples = audioCapturer->getAvailableSamples();
        if (!new_samples.empty())
        {
            audioSamples.insert(audioSamples.end(), new_samples.begin(), new_samples.end());

            // Keep last TIMELINE_SIZE seconds of data
            const size_t max_samples = TIMELINE_SIZE * AudioCapture::SAMPLE_RATE;
            if (audioSamples.size() > max_samples)
            {
                audioSamples.erase(
                    audioSamples.begin(),
                    audioSamples.end() - max_samples);
            }
        }
    }
}

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

void demoTables()
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

        for (int rowIndex = 0; rowIndex < 20; rowIndex++)
        {
            ImGui::TableNextRow();

            srand(rowIndex);

            static float data[100];
            for (int i = 0; i < 100; ++i)
                data[i] = RandomRange(-1.0f, 1.0f);

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("EMG %d", rowIndex);

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.3f V", data[offset]);

            ImGui::TableSetColumnIndex(2);
            ImGui::PushID(rowIndex);

            Sparkline("##spark", data, 100, -1.0f, 1.0f, offset, ImPlot::GetColormapColor(rowIndex), ImVec2(-1, 100));

            ImGui::PopID();
        }
        ImPlot::PopColormap();
        ImGui::EndTable();
    }
}

void widgets()
{
    if (ImGui::Button(capturerManager.isCapturing() ? "Stop Capture" : "Start Capture"))
    {
        toggleAudioCapture(&capturerManager, audioSamples, errorMessage);
    }

    if (ImGui::Button("Clear Display"))
    {
        audioSamples.clear();
    }

    if (!errorMessage.empty())
    {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Error: %s", errorMessage.c_str());
        if (ImGui::Button("Clear Error"))
        {
            errorMessage.clear();
        }
    }
}

int main()
{
    capturerManager.addCapturer(audioCapturer);
    SignalPlotter signalPlotterl(400, AudioCapture::SAMPLE_RATE);

    ImGuiApp app("Audio Visualizer", 1280, 720);
    if (!app.initialize())
        return -1;

    ImPlot::CreateContext();
    app.run(
        [&]()
        {
            widgets();
            getSamples(audioSamples);
            signalPlotterl.plot(audioSamples, TIMELINE_SIZE);
            demoTables();
        });
    ImPlot::DestroyContext();

    return 0;
}
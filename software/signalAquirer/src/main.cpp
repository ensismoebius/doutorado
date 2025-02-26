#include <iostream>
#include <memory>
#include <vector>

#include "../lib/util/imguiGlfw.hpp"
#include "../lib/util/AudioCapture.hpp"
#include "../lib/implot/implot.h"

// Global state for audio capture
std::unique_ptr<AudioCapture> audio_capture;
bool is_capturing = false;
std::vector<float> audio_samples;
std::string error_message;

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
    const float plot_height = 200.0f;
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

            // Keep last 2 seconds of data
            const size_t max_samples = 2 * AudioCapture::SAMPLE_RATE;
            if (audio_samples.size() > max_samples)
            {
                audio_samples.erase(audio_samples.begin(),
                                    audio_samples.end() - max_samples);
            }
        }
    }

    // Show visualization
    audio_visualization();
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
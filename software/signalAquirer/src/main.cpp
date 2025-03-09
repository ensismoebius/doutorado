#include <iostream>
#include <memory>
#include <vector>
#include <random>

#include "../lib/util/CapturerManager.hpp"
#include "../lib/util/AudioCapture.hpp"
#include "../lib/util/MockCapturer.hpp"
#include "../lib/util/SignalPlotter.hpp"
#include "../lib/util/PlotsTable.hpp"

#include "../lib/util/imguiGlfw.hpp"
#include "../lib/implot/implot.h"
#include "main.h"

#define TIMELINE_SIZE 10

using namespace std;

// Signal capture manager and signal capturer
shared_ptr<ICapturer> audioCapturer = make_shared<AudioCapture>();
CapturerManager capturerManager;
PlotsTable plotsTable;

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
            plotsTable.plotAll({audioSamples});
        });
    ImPlot::DestroyContext();

    return 0;
}
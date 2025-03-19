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
std::vector<float> testSignalSamples00;
std::vector<float> testSignalSamples01;

std::string errorMessage;

template <typename T>
inline T RandomRange(T min, T max)
{
    T scale = rand() / (T)RAND_MAX;
    return min + scale * (max - min);
}

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

float getSamples(vector<float> &audioSamplesConteiner)
{
    // Update audio buffer
    const auto new_samples = audioCapturer->getAvailableSamples();
    if (!new_samples.empty())
    {
        audioSamplesConteiner.insert(audioSamplesConteiner.end(), new_samples.begin(), new_samples.end());
        return new_samples.size();
    }

    return 0;
}

float getSamples2(vector<float> &signalSamplesContainer, size_t size)
{
    vector<float> new_samples(size);

    for (auto &item : new_samples)
    {
        item = RandomRange(-0.5f, 0.5f);
    }

    if (!new_samples.empty())
    {
        signalSamplesContainer.insert(signalSamplesContainer.end(), new_samples.begin(), new_samples.end());
        return new_samples.size();
    }
    return 0;
}

float getSamples3(vector<float> &signalSamplesContainer, size_t size)
{
    vector<float> new_samples(size);

    for (size_t i = 0; i < new_samples.size(); i++)
    {
        if (i % 100 == 0)
        {
            new_samples[i] = RandomRange(-0.5f, 0.5f);
        }
    }

    if (!new_samples.empty())
    {
        signalSamplesContainer.insert(signalSamplesContainer.end(), new_samples.begin(), new_samples.end());
        return new_samples.size();
    }
    return 0;
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
        testSignalSamples00.clear();
        testSignalSamples01.clear();
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

vector<float> downsample(vector<float> samples, unsigned int factor)
{
    std::vector<float> downsampled;
    downsampled.reserve(samples.size() / factor);

    for (size_t i = 0; i < samples.size(); i += factor)
    {
        downsampled.push_back(samples[i]);
    }

    return downsampled;
}

int main()
{
    capturerManager.addCapturer(audioCapturer);
    SignalPlotter signalPlotterl(400, AudioCapture::SAMPLE_RATE);

    ImGuiApp app("Audio Visualizer", 1280, 720);
    if (!app.initialize())
        return -1;

    static bool ploting = false;

    ImPlot::CreateContext();
    app.run(
        [&]()
        {
            if (ImGui::Button(ploting ? "Plotando" : "Não plotando"))
            {
                ploting = !ploting;
            }

            widgets();

            if (capturerManager.isCapturing())
            {
                size_t size = getSamples(audioSamples);
                getSamples2(testSignalSamples00, size);
                getSamples3(testSignalSamples01, size);
            }

            if (ploting)
            {
                plotsTable.plotAll({audioSamples, testSignalSamples00, testSignalSamples01}, TIMELINE_SIZE, AudioCapture::SAMPLE_RATE);
            }
        });
    ImPlot::DestroyContext();

    return 0;
}
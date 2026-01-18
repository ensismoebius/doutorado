#include <iostream>
#include <memory>
#include <vector>
#include <random>

#include "../lib/util/CapturerManager.hpp"
#include "../lib/util/SerialCapturer.hpp"
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
shared_ptr<ICapturer> serialCapturer = make_shared<SerialCapturer>("/dev/ttyACM0", B9600);

CapturerManager capturerManager;
PlotsTable plotsTable;

// Global state for audio capture
Signal audioSamples;
Signal serialData;

std::string errorMessage;

template <typename T>
inline T RandomRange(T min, T max)
{
    T scale = rand() / (T)RAND_MAX;
    return min + scale * (max - min);
}

inline void toggleAudioCapture(
    CapturerManager *captureManager,
    Signal samples,
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
        samples.signal.clear();
        return;
    }

    // If NOT capturing and can't start capturing,
    // stores an error message
    errorMessages = captureManager->getErrors();
}

float getSamples(Signal &audioSamplesConteiner)
{

    audioSamplesConteiner.sampleRate = AudioCapture::SAMPLE_RATE;

    // Update audio buffer
    const auto new_samples = audioCapturer->getAvailableSamples();
    if (!new_samples.empty())
    {
        audioSamplesConteiner.signal.insert(audioSamplesConteiner.signal.end(), new_samples.begin(), new_samples.end());

        return new_samples.size();
    }

    return 0;
}

float getSamples2(Signal &serialSamplesConteiner)
{

    serialSamplesConteiner.sampleRate = AudioCapture::SAMPLE_RATE / 40;

    // Update serial buffer
    const auto new_samples = serialCapturer->getAvailableSamples();
    if (!new_samples.empty())
    {
        serialSamplesConteiner.signal.insert(serialSamplesConteiner.signal.end(), new_samples.begin(), new_samples.end());

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
        audioSamples.signal.clear();
        serialData.signal.clear();
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
    capturerManager.addCapturer(serialCapturer);

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
                getSamples2(serialData);
            }

            if (ploting)
            {
                plotsTable.plotAll(
                    {make_pair("audio", audioSamples),
                     make_pair("serial", serialData)},
                    TIMELINE_SIZE);
            }
        });
    ImPlot::DestroyContext();

    return 0;
}
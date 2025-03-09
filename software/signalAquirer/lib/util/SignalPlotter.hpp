#ifndef SIGNAL_PLOTTER_H
#define SIGNAL_PLOTTER_H

#include <vector>
#include "../lib/implot/implot.h"

using namespace std;

class SignalPlotter
{
private:
    float TIME_STEP;
    const float plotHeight;

public:
    SignalPlotter(const float plotHeight, const unsigned int SAMPLE_RATE);
    void plot(std::vector<float> &samples, const float TIMELINE_SIZE);
    ~SignalPlotter();
};

#endif

void SignalPlotter::plot(std::vector<float> &samples, const float TIMELINE_SIZE)
{

    static const float timeStep = this->TIME_STEP;

    // Pass function pointer with correct signature
    static auto getter = [](int idx, void *data) -> ImPlotPoint
    {
        const auto &samples = *static_cast<std::vector<float> *>(data);
        return ImPlotPoint(idx * timeStep, samples[idx]);
    };

    // Configure plot
    if (ImPlot::BeginPlot("Audio Waveform", ImVec2(-1, plotHeight)))
    {
        ImPlot::SetupAxis(ImAxis_X1, "Time (s)");
        ImPlot::SetupAxis(ImAxis_Y1, "Amplitude");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0f, TIMELINE_SIZE, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.2f, 1.2f);
        ImPlot::SetNextLineStyle(ImVec4(0, 1, 0, .5), 1.5f); // Green line, 1.5px thick

        ImPlot::PlotLineG(
            "Audio Signal",
            getter,
            &samples,
            static_cast<int>(samples.size()));

        ImPlot::EndPlot();
    }
}

SignalPlotter::SignalPlotter(
    const float plotHeight,
    const unsigned int SAMPLE_RATE) : TIME_STEP(1.0f / SAMPLE_RATE), plotHeight(plotHeight)
{
}

SignalPlotter::~SignalPlotter()
{
}
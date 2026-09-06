/**
 * @file plotSpikingNetwork.cpp
 * @brief Visualization demo: simulate a tiny LIF network and plot spikes/voltages via ImGui/ImPlot.
 *
 * This demo is meant for intuition-building:
 * - generate a Poisson-like spike input
 * - run a couple of `Lif` neurons forward in time
 * - plot spike trains and membrane potentials
 */

#define DEBUG

#include <iostream>
#include <memory>
#include <vector>

#include "imgui.h"
#include "implot.h"
#include "layers/Layers.hpp"
#include "layers/spiking/Lif.hpp"
#include "utility/imgui_glfw.hpp"
#include "utility/synthetic_spike_data.hpp"

using ImGui::Begin;
using ImGui::End;

using ImPlot::BeginPlot;
using ImPlot::DestroyContext;
using ImPlot::EndPlot;
using ImPlot::PlotLine;
using ImPlot::PlotScatter;
using ImPlot::SetupAxes;

using ImPlot::CreateContext;
using nn::Lif;
using std::cerr;
using std::to_string;
using std::vector;

constexpr int n_steps = 200;
constexpr float max_rate = 0.5F;
constexpr float delta_t = 1.0F;
constexpr float resistence = 3.0F;
constexpr float capacitance = 2.0F;
constexpr float v_threshold = 1.0F;

namespace
{

// Per-timestep series collected while simulating the 1-1-1 network, used only for plotting.
struct SimulationResult
{
    vector<float> plot_input_spikes;
    vector<float> plot_hidden_spikes;
    vector<float> plot_output_spikes;
    vector<float> plot_hidden_vmems;
    vector<float> plot_output_vmems;
};

// Runs the input spike train through hidden_neuron -> output_neuron for n_steps, recording
// each neuron's spike output and membrane potential at every step.
auto run_simulation(Lif& hidden_neuron, Lif& output_neuron, const vector<nn::Tensor>& spike_inputs)
    -> SimulationResult
{
    SimulationResult sim;

    for (int t = 0; t < n_steps; ++t)
    {
        // Input spike at this timestep
        nn::Tensor in_tensor = spike_inputs[t];

        // Forward through the network
        auto hidden_out = hidden_neuron.forward(in_tensor);
        auto output_out = output_neuron.forward(hidden_out);

        // Store data for plotting
        sim.plot_input_spikes.push_back(in_tensor(0, 0));
        sim.plot_hidden_spikes.push_back(hidden_out(0, 0));
        sim.plot_output_spikes.push_back(output_out(0, 0));

        // Membrane potentials
        sim.plot_hidden_vmems.push_back(hidden_neuron.v_mem(0, 0));
        sim.plot_output_vmems.push_back(output_neuron.v_mem(0, 0));
    }

    return sim;
}

// Collects the timestep indices where `series[t]` crosses the spike threshold (> 0.5).
auto collect_spike_times(const vector<float>& series) -> vector<float>
{
    vector<float> times;
    for (int t = 0; t < n_steps; ++t)
    {
        if (series[t] > 0.5F)
        {
            times.push_back((float) t);
        }
    }
    return times;
}

// Draws the input/hidden/output spike raster (one scatter row per neuron).
void draw_spike_raster_plot(const SimulationResult& sim)
{
    if (BeginPlot("Spike Raster Plot", ImVec2(-1, 300)))
    {
        SetupAxes("Time", "Neuron");

        // Input neuron
        vector<float> input_spike_times = collect_spike_times(sim.plot_input_spikes);
        if (!input_spike_times.empty())
        {
            vector<float> y(input_spike_times.size(), 0.0F);
            PlotScatter(
                "Input", input_spike_times.data(), y.data(), (int) input_spike_times.size());
        }

        // Hidden neuron
        vector<float> hidden_spike_times = collect_spike_times(sim.plot_hidden_spikes);
        if (!hidden_spike_times.empty())
        {
            vector<float> y(hidden_spike_times.size(), 1.0F);
            PlotScatter(
                "Hidden", hidden_spike_times.data(), y.data(), (int) hidden_spike_times.size());
        }

        // Output neuron
        vector<float> output_spike_times = collect_spike_times(sim.plot_output_spikes);
        if (!output_spike_times.empty())
        {
            vector<float> y(output_spike_times.size(), 2.0F);
            PlotScatter(
                "Output", output_spike_times.data(), y.data(), (int) output_spike_times.size());
        }

        EndPlot();
    }
}

// Draws the hidden/output membrane potential traces.
void draw_membrane_potential_plot(const SimulationResult& sim)
{
    if (BeginPlot("Membrane Potential", ImVec2(-1, 300)))
    {
        SetupAxes("Time", "V_mem");
        PlotLine("Hidden V_mem", sim.plot_hidden_vmems.data(), n_steps);
        PlotLine("Output V_mem", sim.plot_output_vmems.data(), n_steps);
        EndPlot();
    }
}

} // namespace

auto main() -> int
{
    try
    {
        // Generate single input spike train (Poisson)
        auto [spike_inputs, _] = generate_autoencoder_spike_data(1, 1, n_steps, max_rate, delta_t);

        // Setup 1 hidden LIF neuron and 1 output neuron (LIF)
        Lif hidden_neuron(delta_t,                       // dt
            resistence,                                  // R
            capacitance,                                 // C
            v_threshold,                                 // reset to zero or subtract threshold
            true,                                        // reset to zero
            0.0F,                                        // reset potential value
            std::make_shared<ExponentialSurrogate>(0.5F) // surrogate gradient
        );
        Lif output_neuron(delta_t,                       // dt
            resistence,                                  // R
            capacitance,                                 // C
            v_threshold,                                 // reset to zero or subtract threshold
            true,                                        // reset to zero
            0.0F,                                        // reset potential value
            std::make_shared<ExponentialSurrogate>(0.5F) // surrogate gradient
        );

        SimulationResult sim = run_simulation(hidden_neuron, output_neuron, spike_inputs);

        // Visualization
        ImGuiApp window("1-1-1 Spiking Network Visualization", 1200, 800);
        if (!window.initialize())
        {
            cerr << "Failed to initialize ImGuiApp" << '\n';
            return 1;
        }

        CreateContext();

        window.run(
            [&]()
            {
                Begin("Neuron Output");
                draw_spike_raster_plot(sim);
                draw_membrane_potential_plot(sim);
                End();
            });
        DestroyContext();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred." << std::endl;
        return 1;
    }
}
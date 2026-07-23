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

        // Data for plotting
        vector<float> plot_input_spikes;

        // Neuron outputs
        vector<float> plot_hidden_spikes;
        vector<float> plot_output_spikes;

        // Membrane potentials
        vector<float> plot_hidden_vmems;
        vector<float> plot_output_vmems;

        // Simulate
        for (int t = 0; t < n_steps; ++t)
        {
            // Input spike at this timestep
            nn::Tensor in_tensor = spike_inputs[t];

            // Forward through the network
            auto hidden_out = hidden_neuron.forward(in_tensor);
            auto output_out = output_neuron.forward(hidden_out);

            // Store data for plotting
            plot_input_spikes.push_back(in_tensor(0, 0));
            plot_hidden_spikes.push_back(hidden_out(0, 0));
            plot_output_spikes.push_back(output_out(0, 0));

            // Membrane potentials
            plot_hidden_vmems.push_back(hidden_neuron.v_mem(0, 0));
            plot_output_vmems.push_back(output_neuron.v_mem(0, 0));
        }

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

                // Spike raster plot
                if (BeginPlot("Spike Raster Plot", ImVec2(-1, 300)))
                {
                    SetupAxes("Time", "Neuron");

                    // Input neuron
                    vector<float> input_spike_times;
                    for (int t = 0; t < n_steps; ++t)
                    {
                        if (plot_input_spikes[t] > 0.5F)
                        {
                            input_spike_times.push_back((float) t);
                        }
                    }
                    if (!input_spike_times.empty())
                    {
                        vector<float> y(input_spike_times.size(), 0.0F);
                        PlotScatter("Input",
                            input_spike_times.data(),
                            y.data(),
                            (int) input_spike_times.size());
                    }

                    // Hidden neuron
                    vector<float> hidden_spike_times;
                    for (int t = 0; t < n_steps; ++t)
                    {
                        if (plot_hidden_spikes[t] > 0.5F)
                        {
                            hidden_spike_times.push_back((float) t);
                        }
                    }
                    if (!hidden_spike_times.empty())
                    {
                        vector<float> y(hidden_spike_times.size(), 1.0F);
                        PlotScatter("Hidden",
                            hidden_spike_times.data(),
                            y.data(),
                            (int) hidden_spike_times.size());
                    }

                    // Output neuron
                    vector<float> output_spike_times;
                    for (int t = 0; t < n_steps; ++t)
                    {
                        if (plot_output_spikes[t] > 0.5F)
                        {
                            output_spike_times.push_back((float) t);
                        }
                    }
                    if (!output_spike_times.empty())
                    {
                        vector<float> y(output_spike_times.size(), 2.0F);
                        PlotScatter("Output",
                            output_spike_times.data(),
                            y.data(),
                            (int) output_spike_times.size());
                    }

                    EndPlot();
                }

                // Membrane potential plot
                if (BeginPlot("Membrane Potential", ImVec2(-1, 300)))
                {
                    SetupAxes("Time", "V_mem");
                    PlotLine("Hidden V_mem", plot_hidden_vmems.data(), n_steps);
                    PlotLine("Output V_mem", plot_output_vmems.data(), n_steps);
                    EndPlot();
                }
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
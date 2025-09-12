#include "imgui.h"
#include "implot.h"
#include "layers/Leaky.hpp"
#include "util/imguiGlfw.hpp"
#include "util/synthetic_spike_data.hpp"
#include <iostream>
#include <vector>

using ImGui::Begin;
using ImGui::End;
using ImPlot::BeginPlot;
using ImPlot::EndPlot;
using ImPlot::PlotLine;
using ImPlot::PlotScatter;
using std::cerr;
using std::vector;

auto main() -> int {
  constexpr int n_neurons = 8;
  constexpr int n_steps = 200;
  constexpr float max_rate = 0.5F;
  constexpr float delta_t = 1.0F;

  // Generate synthetic spike input (Poisson)
  auto [spike_inputs, _] =
      generate_autoencoder_spike_data(1, n_neurons, n_steps, max_rate, delta_t);

  // Setup LIF neurons
  std::vector<Leaky> neurons;
  neurons.reserve(n_neurons);
  for (int i = 0; i < n_neurons; ++i) {
    neurons.emplace_back(delta_t, 5.0F, 1.0F, 1.0F, true, 0.0F, 1.0F);
  }

  // Data for plotting
  vector<std::vector<float>> spikes(n_neurons);
  vector<std::vector<float>> vmems(n_neurons);

  // Simulate
  for (int t = 0; t < n_steps; ++t) {
    const auto &input = spike_inputs[t].data;
    for (int n_index = 0; n_index < n_neurons; ++n_index) {
      // Each neuron gets its own input channel
      Tensor single_input = Tensor(1, 1);
      single_input.data(0, 0) = input(0, n_index);
      auto out = neurons[n_index].forward(single_input);
      spikes[n_index].push_back(out.data(0, 0));
      vmems[n_index].push_back(neurons[n_index].v_mem(0, 0));
    }
  }

  // Visualization
  ImGuiApp app("Spiking Neuron Output Visualization", 1200, 800);
  if (!app.initialize()) {
    cerr << "Failed to initialize ImGuiApp" << '\n';
    return 1;
  }
  ImPlot::CreateContext();
  app.run([&]() {
    Begin("Neuron Output");
    if (ImPlot::BeginPlot("Spike Raster Plot", ImVec2(-1, 300))) {
      ImPlot::SetupAxes("Time", "Neuron");
      for (int n = 0; n < n_neurons; ++n) {
        vector<float> spike_times;
        for (int t = 0; t < n_steps; ++t) {
          if (spikes[n][t] > 0.5F) {
            spike_times.push_back((float)t);
          }
        }
        if (!spike_times.empty()) {
          vector<float> y(spike_times.size(), (float)n);
          PlotScatter(("Neuron " + std::to_string(n)).c_str(), spike_times.data(), y.data(),
                      (int)spike_times.size());
        }
      }
      EndPlot();
    }

    if (ImPlot::BeginPlot("Membrane Potential", ImVec2(-1, 300))) {
      ImPlot::SetupAxes("Time", "V_mem");
      for (int n_index = 0; n_index < n_neurons; ++n_index) {
        PlotLine(("Neuron " + std::to_string(n_index)).c_str(), vmems[n_index].data(), n_steps);
      }
      EndPlot();
    }
    End();
  });
  ImPlot::DestroyContext();
  return 0;
}
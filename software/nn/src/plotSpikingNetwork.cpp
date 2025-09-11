#include "imgui.h"
#include "implot.h"
#include "layers/Leaky.hpp"
#include "util/imguiGlfw.hpp"
#include "util/synthetic_spike_data.hpp"
#include <iostream>
#include <vector>

auto main() -> int {
  constexpr int n_neurons = 8;
  constexpr int n_steps = 200;
  constexpr float max_rate = 0.5f;
  constexpr float dt = 1.0f;

  // Generate synthetic spike input (Poisson)
  auto [spike_inputs, _] = generate_autoencoder_spike_data(1, n_neurons, n_steps, max_rate, dt);

  // Setup LIF neurons
  std::vector<Leaky> neurons;
  for (int i = 0; i < n_neurons; ++i) {
    neurons.emplace_back(dt, 5.0f, 1.0f, 1.0f, true, 0.0f, 1.0f);
  }

  // Data for plotting
  std::vector<std::vector<float>> spikes(n_neurons);
  std::vector<std::vector<float>> vmems(n_neurons);

  // Simulate
  for (int t = 0; t < n_steps; ++t) {
    const auto &input = spike_inputs[t].data;
    for (int n = 0; n < n_neurons; ++n) {
      // Each neuron gets its own input channel
      Tensor single_input = Tensor(1, 1);
      single_input.data(0, 0) = input(0, n);
      auto out = neurons[n].forward(single_input);
      spikes[n].push_back(out.data(0, 0));
      vmems[n].push_back(neurons[n].v_mem(0, 0));
    }
  }

  // Visualization
  ImGuiApp app("Spiking Neuron Output Visualization", 1200, 800);
  if (!app.initialize()) {
    std::cerr << "Failed to initialize ImGuiApp" << std::endl;
    return 1;
  }
  ImPlot::CreateContext();
  app.run([&]() {
    ImGui::Begin("Neuron Output");
    if (ImPlot::BeginPlot("Spike Raster Plot", "Time", "Neuron", ImVec2(-1, 300))) {
      for (int n = 0; n < n_neurons; ++n) {
        std::vector<float> spike_times;
        for (int t = 0; t < n_steps; ++t) {
          if (spikes[n][t] > 0.5f)
            spike_times.push_back((float)t);
        }
        if (!spike_times.empty()) {
          std::vector<float> y(spike_times.size(), (float)n);
          ImPlot::PlotScatter(("Neuron " + std::to_string(n)).c_str(), spike_times.data(), y.data(),
                              (int)spike_times.size());
        }
      }
      ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("Membrane Potential", "Time", "V_mem", ImVec2(-1, 300))) {
      for (int n = 0; n < n_neurons; ++n) {
        ImPlot::PlotLine(("Neuron " + std::to_string(n)).c_str(), vmems[n].data(), n_steps);
      }
      ImPlot::EndPlot();
    }
    ImGui::End();
  });
  ImPlot::DestroyContext();
  return 0;
}
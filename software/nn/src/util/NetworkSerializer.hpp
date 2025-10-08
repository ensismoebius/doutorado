#pragma once

#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../layers/Leaky.hpp"
#include "../layers/LeakyReLU.hpp"
#include "../layers/Linear.hpp"
#include "../layers/ReLU.hpp"
#include "../layers/Sequential.hpp"
#include <cnpy.h>

using cnpy::NpyArray;
using cnpy::npz_load;
using cnpy::npz_save;
using cnpy::npz_t;
using Eigen::Index;
using Eigen::Map;
using Eigen::MatrixXf;
using std::cerr;
using std::cout;
using std::dynamic_pointer_cast;
using std::exception;
using std::make_shared;
using std::map;
using std::pair;
using std::runtime_error;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::vector;
using std::filesystem::create_directories;
using std::filesystem::exists;
using std::filesystem::path;

/**
 * @brief NetworkSerializer class to save and load neural network weights, biases, and architecture
 *        in a PyTorch-compatible format. Now supports full model serialization (structure +
 * parameters) for Linear, Leaky, LeakyReLU, and ReLU layers.
 */
class NetworkSerializer {
private:
  static constexpr const char *WEIGHTS_SUFFIX = ".weight";
  static constexpr const char *BIAS_SUFFIX = ".bias";

  // Structure to hold parameter metadata (similar to PyTorch's state_dict)
  struct ParameterInfo {
    string name;          // Full parameter name (e.g., "layer.0.weight")
    string type;          // Parameter type (e.g., "Linear")
    vector<size_t> shape; // Parameter shape
    const float *data;    // Pointer to parameter data
  };

  /**
   * @brief Get the type name of a module
   * @param module The module to get the type for
   * @return The type name as a string
   */
  static auto getModuleType(const shared_ptr<Module> &module) -> string {
    if (dynamic_pointer_cast<Linear>(module)) {
      return "Linear";
    }
    // Add more module types here as needed
    return "Unknown";
  }

  /**
   * @brief Add a layer's weights and biases to the data map
   *
   * @param layer The layer to save
   * @param name Layer name for array naming
   */
  static auto collectStateDict(const Sequential &model) -> vector<ParameterInfo> {
    vector<ParameterInfo> state_dict;
    size_t linearLayerCount = 0;

    size_t layerCount = 0;
    for (const auto &layer : model.layers) {
      if (auto linearLayer = dynamic_pointer_cast<Linear>(layer)) {
        const string modulePath = to_string(layerCount);
        state_dict.push_back({modulePath + WEIGHTS_SUFFIX,
                              "Linear",
                              {static_cast<size_t>(linearLayer->weight.data.rows()),
                               static_cast<size_t>(linearLayer->weight.data.cols())},
                              linearLayer->weight.data.data()});
        size_t out_features = static_cast<size_t>(linearLayer->bias.data.rows());
        std::vector<float> bias_1d(out_features);
        for (size_t i = 0; i < out_features; ++i) {
          bias_1d[i] = linearLayer->bias.data(static_cast<Eigen::Index>(i), 0);
        }
        state_dict.push_back({modulePath + BIAS_SUFFIX, "Linear", {out_features}, bias_1d.data()});
      } else if (auto leakyLayer = dynamic_pointer_cast<Leaky>(layer)) {
        const string modulePath = to_string(layerCount);
        // Save resistance and voltage_threshold as parameters
        state_dict.push_back({modulePath + ".resistance",
                              "Leaky",
                              {static_cast<size_t>(leakyLayer->resistance.data.rows()),
                               static_cast<size_t>(leakyLayer->resistance.data.cols())},
                              leakyLayer->resistance.data.data()});
        state_dict.push_back({modulePath + ".voltage_threshold",
                              "Leaky",
                              {static_cast<size_t>(leakyLayer->voltage_threshold.data.rows()),
                               static_cast<size_t>(leakyLayer->voltage_threshold.data.cols())},
                              leakyLayer->voltage_threshold.data.data()});
        // No trainable params for dt, capacitance, reset_zero, reset_potential, surrogate_gradient
      }
      // LeakyReLU and ReLU have no trainable params
      layerCount++;
    }
    return state_dict;
  }

  /**
   * @brief Load a single layer's weights and biases from npz data
   *
   * @param layer The layer to load into
   * @param name Layer name for array naming
   * @param data The loaded npz data
   */
  static void loadLayer(const shared_ptr<Linear> &layer, const string &modulePath,
                        const npz_t &data) {

    // Load weights - PyTorch format: [out_features, in_features]
    string weight_name = modulePath + WEIGHTS_SUFFIX;
    auto w_it = data.find(weight_name);
    if (w_it == data.end()) {
      throw runtime_error("Weight array not found for module: " + modulePath);
    }
    const NpyArray &arr_w = w_it->second;
    const auto *weight_data = arr_w.data<float>();
    // Map the data maintaining PyTorch's layout
    layer->weight.data =
        Map<const MatrixXf>(weight_data, layer->weight.data.rows(), layer->weight.data.cols());

    // Load bias - PyTorch format: [out_features]
    string bias_name = modulePath + BIAS_SUFFIX;
    auto b_it = data.find(bias_name);
    if (b_it == data.end()) {
      throw runtime_error("Bias array not found for module: " + modulePath);
    }

    const NpyArray &arr_b = b_it->second;
    const auto *bias_data = arr_b.data<float>();
    layer->bias.data =
        Map<const MatrixXf>(bias_data, layer->bias.data.rows(), layer->bias.data.cols());
  }

public:
  /**
   * @brief Save the full architecture and parameters of a Sequential model to a single npz file
   *
   * The architecture (layer types, order, and configuration) is saved as metadata, allowing full
   * reconstruction. Currently supports:
   *   - Linear: weights, bias
   *   - Leaky: resistance, voltage_threshold, dt, capacitance, reset_zero, reset_potential
   *   - LeakyReLU: alpha
   *   - ReLU: stateless
   * Extend as needed for other layer types.
   *
   * @param model The Sequential model to save
   * @param safe_filepath Path to the .npz file to save
   * @return true if save was successful, false otherwise
   */
  static auto saveNetwork(const Sequential &model, const string &safe_filepath) -> bool {
    try {
      // Create the directory if it doesn't exist
      create_directories(path(safe_filepath).parent_path());

      // Collect state dictionary
      auto state_dict = collectStateDict(model);

      // Save architecture metadata as a single string (lines separated by '\n')
      std::string arch_metadata_str;
      for (const auto &layer : model.layers) {
        if (auto linearLayer = dynamic_pointer_cast<Linear>(layer)) {
          arch_metadata_str += "Linear:" + std::to_string(linearLayer->weight.data.cols()) + ":" +
                               std::to_string(linearLayer->weight.data.rows()) + "\n";
        } else if (auto leakyLayer = dynamic_pointer_cast<Leaky>(layer)) {
          arch_metadata_str += "Leaky:" + std::to_string(leakyLayer->dt) + ":" +
                               std::to_string(leakyLayer->resistance.data(0, 0)) + ":" +
                               std::to_string(leakyLayer->capacitance) + ":" +
                               std::to_string(leakyLayer->voltage_threshold.data(0, 0)) + ":" +
                               (leakyLayer->reset_zero ? "1" : "0") + ":" +
                               std::to_string(leakyLayer->reset_potential) + "\n";
        } else if (auto leakyReLULayer = dynamic_pointer_cast<LeakyReLU>(layer)) {
          arch_metadata_str += "LeakyReLU:" + std::to_string(leakyReLULayer->alpha) + "\n";
        } else if (dynamic_pointer_cast<ReLU>(layer)) {
          arch_metadata_str += "ReLU\n";
        }
      }
      // Convert to char array for npz_save
      std::vector<char> arch_metadata_vec(arch_metadata_str.begin(), arch_metadata_str.end());
      npz_save(safe_filepath, "__architecture__", arch_metadata_vec.data(),
               {arch_metadata_vec.size()}, "w");

      // Save parameters from state dictionary
      // Special handling for bias: if param.shape.size() == 1, treat as 1D
      for (const auto &param : state_dict) {
        npz_save(safe_filepath, param.name, param.data, param.shape, "a");
      }

      cout << "Successfully saved network to file: " << safe_filepath << "\n";
      return true;

    } catch (const exception &e) {
      cerr << "Error saving network: " << e.what() << "\n";
      return false;
    }
  }

  /**
   * @brief Load the full architecture and parameters from a single npz file into a Sequential model
   *
   * The architecture (layer types, order, and configuration) is read from metadata and the model is
   * reconstructed. Currently supports:
   *   - Linear: weights, bias
   *   - Leaky: resistance, voltage_threshold, dt, capacitance, reset_zero, reset_potential
   *   - LeakyReLU: alpha
   *   - ReLU: stateless
   * Extend as needed for other layer types.
   *
   * @param model The Sequential model to load into (will be cleared and rebuilt)
   * @param safe_filepath Path to the .npz file to load
   * @return true if load was successful, false otherwise
   */
  static auto loadNetwork(Sequential &model, const string &safe_filepath) -> bool {
    try {
      if (!exists(safe_filepath)) {
        throw runtime_error("Network file does not exist: " + safe_filepath);
      }

      // Load the npz file
      npz_t data = npz_load(safe_filepath);

      // --- Architecture deserialization ---
      // Read architecture metadata string
      auto arch_it = data.find("__architecture__");
      if (arch_it == data.end()) {
        throw runtime_error("Architecture metadata not found in file: " + safe_filepath);
      }
      const NpyArray &arch_arr = arch_it->second;
      std::string arch_str(arch_arr.data<char>(), arch_arr.data<char>() + arch_arr.shape[0]);

      // Parse architecture lines
      std::vector<std::string> arch_lines;
      size_t start = 0;
      size_t end = 0;
      while ((end = arch_str.find('\n', start)) != std::string::npos) {
        if (end > start) {
          arch_lines.push_back(arch_str.substr(start, end - start));
        }
        start = end + 1;
      }

      // Reconstruct model.layers according to architecture
      model.layers.clear();
      for (const auto &line : arch_lines) {
        if (line.rfind("Linear:", 0) == 0) {
          size_t pos1 = line.find(":");
          size_t pos2 = line.find(":", pos1 + 1);
          int in_features = std::stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
          int out_features = std::stoi(line.substr(pos2 + 1));
          model.layers.push_back(std::make_shared<Linear>(in_features, out_features));
        } else if (line.rfind("Leaky:", 0) == 0) {
          // Format: Leaky:dt:R:C:Vth:reset_zero:reset_potential
          std::vector<std::string> tokens;
          size_t prev = 0, pos = 0;
          while ((pos = line.find(":", prev)) != std::string::npos) {
            tokens.push_back(line.substr(prev, pos - prev));
            prev = pos + 1;
          }
          tokens.push_back(line.substr(prev));
          if (tokens.size() == 7) {
            float dt = std::stof(tokens[1]);
            float R = std::stof(tokens[2]);
            float C = std::stof(tokens[3]);
            float Vth = std::stof(tokens[4]);
            bool reset_zero = (tokens[5] == "1");
            float reset_potential = std::stof(tokens[6]);
            model.layers.push_back(
                std::make_shared<Leaky>(dt, R, C, Vth, reset_zero, reset_potential));
          }
        } else if (line.rfind("LeakyReLU:", 0) == 0) {
          // Format: LeakyReLU:alpha
          size_t pos1 = line.find(":");
          float alpha = std::stof(line.substr(pos1 + 1));
          model.layers.push_back(std::make_shared<LeakyReLU>(alpha));
        } else if (line == "ReLU") {
          model.layers.push_back(std::make_shared<ReLU>());
        }
      }

      // Create a mapping of parameter names to their data
      map<string, pair<const float *, vector<size_t>>> param_map;
      for (const auto &[name, array] : data) {
        // Only float arrays (skip metadata)
        if (name == "__architecture__") {
          continue;
        }
        const auto *param_data = array.data<float>();
        param_map[name] = {param_data, array.shape};
      }

      // Load parameters into layers
      size_t layerCount = 0;
      for (const auto &layer : model.layers) {
        const string modulePath = to_string(layerCount);
        if (auto linearLayer = dynamic_pointer_cast<Linear>(layer)) {
          // Load weights
          string weight_name = modulePath + WEIGHTS_SUFFIX;
          auto weight_it = param_map.find(weight_name);
          if (weight_it == param_map.end()) {
            throw runtime_error("Weight array not found for module: " + modulePath);
          }
          const auto &[weight_data, weight_shape] = weight_it->second;
          linearLayer->weight.data =
              Map<const MatrixXf>(weight_data, static_cast<Index>(weight_shape[0]),
                                  static_cast<Index>(weight_shape[1]));

          // Load bias (expect 1D [out_features])
          string bias_name = modulePath + BIAS_SUFFIX;
          auto bias_it = param_map.find(bias_name);
          if (bias_it == param_map.end()) {
            throw runtime_error("Bias array not found for module: " + modulePath);
          }
          const auto &[bias_data, bias_shape] = bias_it->second;
          if (bias_shape.size() == 1) {
            for (Index i = 0; i < static_cast<Index>(bias_shape[0]); ++i) {
              linearLayer->bias.data(i, 0) = bias_data[i];
            }
          } else {
            linearLayer->bias.data = Map<const MatrixXf>(
                bias_data, static_cast<Index>(bias_shape[0]), static_cast<Index>(bias_shape[1]));
          }
        } else if (auto leakyLayer = dynamic_pointer_cast<Leaky>(layer)) {
          // Load resistance
          string resistance_name = modulePath + ".resistance";
          auto r_it = param_map.find(resistance_name);
          if (r_it == param_map.end()) {
            throw runtime_error("Resistance array not found for module: " + modulePath);
          }
          const auto &[r_data, r_shape] = r_it->second;
          leakyLayer->resistance.data = Map<const MatrixXf>(r_data, static_cast<Index>(r_shape[0]),
                                                            static_cast<Index>(r_shape[1]));
          // Load voltage_threshold
          string vth_name = modulePath + ".voltage_threshold";
          auto vth_it = param_map.find(vth_name);
          if (vth_it == param_map.end()) {
            throw runtime_error("Voltage threshold array not found for module: " + modulePath);
          }
          const auto &[vth_data, vth_shape] = vth_it->second;
          leakyLayer->voltage_threshold.data = Map<const MatrixXf>(
              vth_data, static_cast<Index>(vth_shape[0]), static_cast<Index>(vth_shape[1]));
        }
        // LeakyReLU and ReLU have no trainable params to restore
        layerCount++;
      }

      cout << "Successfully loaded network from file: " << safe_filepath << "\n";
      return true;

    } catch (const exception &e) {
      cerr << "Error loading network: " << e.what() << "\n";
      return false;
    }
  }
};
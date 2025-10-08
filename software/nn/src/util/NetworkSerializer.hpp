#pragma once

#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../layers/Linear.hpp"
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
 * parameters).
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

    for (const auto &layer : model.layers) {
      if (auto linearLayer = dynamic_pointer_cast<Linear>(layer)) {
        // Use PyTorch standard sequential numbering
        const string modulePath = to_string(linearLayerCount);

        // Add weights
        state_dict.push_back({modulePath + WEIGHTS_SUFFIX,
                              "Linear",
                              {static_cast<size_t>(linearLayer->weight.data.rows()),
                               static_cast<size_t>(linearLayer->weight.data.cols())},
                              linearLayer->weight.data.data()});

        // Add bias
        state_dict.push_back({modulePath + BIAS_SUFFIX,
                              "Linear",
                              {static_cast<size_t>(linearLayer->bias.data.rows()),
                               static_cast<size_t>(linearLayer->bias.data.cols())},
                              linearLayer->bias.data.data()});

        linearLayerCount++;
      }
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
   * reconstruction. Currently supports Linear layers. Extend as needed for other layer types.
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
        }
        // Add more layer types here as needed
      }
      // Convert to char array for npz_save
      std::vector<char> arch_metadata_vec(arch_metadata_str.begin(), arch_metadata_str.end());
      npz_save(safe_filepath, "__architecture__", arch_metadata_vec.data(),
               {arch_metadata_vec.size()}, "w");

      // Save parameters from state dictionary
      for (const auto &param : state_dict) {
        // Save parameter data
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
   * reconstructed. Currently supports Linear layers. Extend as needed for other layer types.
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
        // Format: "Linear:in_features:out_features"
        if (line.rfind("Linear:", 0) == 0) {
          size_t pos1 = line.find(":");
          size_t pos2 = line.find(":", pos1 + 1);
          int in_features = std::stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
          int out_features = std::stoi(line.substr(pos2 + 1));
          model.layers.push_back(std::make_shared<Linear>(in_features, out_features));
        }
        // Add more layer types here as needed
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
      size_t linearLayerCount = 0;
      for (const auto &layer : model.layers) {
        if (auto linearLayer = dynamic_pointer_cast<Linear>(layer)) {
          // Use PyTorch standard sequential numbering
          const string modulePath = to_string(linearLayerCount);

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

          // Load bias
          string bias_name = modulePath + BIAS_SUFFIX;
          auto bias_it = param_map.find(bias_name);
          if (bias_it == param_map.end()) {
            throw runtime_error("Bias array not found for module: " + modulePath);
          }
          const auto &[bias_data, bias_shape] = bias_it->second;
          linearLayer->bias.data = Map<const MatrixXf>(bias_data, static_cast<Index>(bias_shape[0]),
                                                       static_cast<Index>(bias_shape[1]));

          linearLayerCount++;
        }
      }

      cout << "Successfully loaded network from file: " << safe_filepath << "\n";
      return true;

    } catch (const exception &e) {
      cerr << "Error loading network: " << e.what() << "\n";
      return false;
    }
  }
};
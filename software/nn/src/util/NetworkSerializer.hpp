#pragma once

#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
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
using std::stof;
using std::stoi;
using std::string;
using std::to_string;
using std::vector;
using std::filesystem::create_directories;
using std::filesystem::exists;
using std::filesystem::path;

/**
 * @brief Serializes and deserializes Sequential models, including architecture and parameters.
 *
 * This class saves and loads the entire state of a Sequential model to a .npz file.
 * It stores the architecture (layer types and their configurations) and all trainable
 * parameters for the following layer types:
 * - Linear
 * - LeakyReLU
 * - Leaky (LIF neuron)
 * - ReLU
 */
class NetworkSerializer {
private:
  static constexpr const char *WEIGHTS_SUFFIX = ".weight";
  static constexpr const char *BIAS_SUFFIX = ".bias";

  // Enum to represent layer types for the switch-based implementation.
  enum class LayerType { Linear, LeakyReLU, Leaky, ReLU, Unknown };

  /**
   * @brief Determines the type of a layer module.
   * @param layer A shared pointer to the layer.
   * @return The corresponding LayerType enum value.
   */
  static auto getLayerType(const shared_ptr<Module> &layer) -> LayerType {
    if (dynamic_pointer_cast<Linear>(layer)) {
      return LayerType::Linear;
    }
    if (dynamic_pointer_cast<LeakyReLU>(layer)) {
      return LayerType::LeakyReLU;
    }
    if (dynamic_pointer_cast<Leaky>(layer)) {
      return LayerType::Leaky;
    }
    if (dynamic_pointer_cast<ReLU>(layer)) {
      return LayerType::ReLU;
    }
    return LayerType::Unknown;
  }

  // --- Save Handlers ---

  /**
   * @brief Handles serialization for a single Linear layer.
   */
  static void _saveLinear(const shared_ptr<Linear> &layer, size_t index, string &arch_str,
                          map<string, pair<vector<size_t>, const float *>> &params) {
    arch_str +=
        "Linear:" + to_string(layer->in_features) + ":" + to_string(layer->out_features) + "\n";

    params[to_string(index) + WEIGHTS_SUFFIX] = {
        {(size_t)layer->weight.data.rows(), (size_t)layer->weight.data.cols()},
        layer->weight.data.data()};

    params[to_string(index) + BIAS_SUFFIX] = {{(size_t)layer->bias.data.rows()},
                                              layer->bias.data.data()};
  }

  /**
   * @brief Handles serialization for a single LeakyReLU layer.
   */
  static void _saveLeakyReLU(const shared_ptr<LeakyReLU> &layer, string &arch_str) {
    arch_str += "LeakyReLU:" + to_string(layer->alpha) + "\n";
  }

  /**
   * @brief Handles serialization for a single ReLU layer.
   */
  static void _saveReLU(string &arch_str) {
    arch_str += "ReLU\n";
  }

  /**
   * @brief Handles serialization for a single Leaky (LIF) layer.
   */
  static void _saveLeaky(const shared_ptr<Leaky> &layer, size_t index, string &arch_str,
                         map<string, pair<vector<size_t>, const float *>> &params) {
    arch_str += "Leaky:" + to_string(layer->dt) + ":" + to_string(layer->resistance.data(0, 0)) +
                ":" + to_string(layer->capacitance) + ":" +
                to_string(layer->voltage_threshold.data(0, 0)) + ":" +
                (layer->reset_zero ? "1" : "0") + ":" + to_string(layer->reset_potential) + "\n";
    params[to_string(index) + ".resistance"] = {
        {(size_t)layer->resistance.data.rows(), (size_t)layer->resistance.data.cols()},
        layer->resistance.data.data()};
    params[to_string(index) + ".voltage_threshold"] = {
        {(size_t)layer->voltage_threshold.data.rows(),
         (size_t)layer->voltage_threshold.data.cols()},
        layer->voltage_threshold.data.data()};
  }

  // --- Load Handlers ---

  /**
   * @brief Handles parameter loading for a single Linear layer.
   */
  static void _loadLinearParams(shared_ptr<Linear> layer, size_t index, const npz_t &data) {
    string weight_name = to_string(index) + WEIGHTS_SUFFIX;
    auto w_it = data.find(weight_name);
    if (w_it == data.end())
      throw runtime_error("Weight array not found for module: " + to_string(index));
    const NpyArray &arr_w = w_it->second;
    layer->weight.data = Map<const MatrixXf>(arr_w.data<float>(), layer->weight.data.rows(),
                                             layer->weight.data.cols());

    string bias_name = to_string(index) + BIAS_SUFFIX;
    auto b_it = data.find(bias_name);
    if (b_it == data.end()) {
      throw runtime_error("Bias array not found for module: " + to_string(index));
    }
    const NpyArray &arr_b = b_it->second;
    const auto *bias_data = arr_b.data<float>();

    if (arr_b.shape.size() == 1) { // Handle 1D bias array
      for (Index i = 0; i < static_cast<Index>(arr_b.shape[0]); ++i) {
        layer->bias.data(i, 0) = bias_data[i];
      }
    } else { // Handle 2D bias array
      layer->bias.data =
          Map<const MatrixXf>(bias_data, layer->bias.data.rows(), layer->bias.data.cols());
    }
  }

  /**
   * @brief Handles parameter loading for a single Leaky (LIF) layer.
   */
  static void _loadLeakyParams(shared_ptr<Leaky> layer, size_t index, const npz_t &data) {
    string res_name = to_string(index) + ".resistance";
    auto r_it = data.find(res_name);
    if (r_it == data.end()) {
      throw runtime_error("Resistance array not found for module: " + to_string(index));
    }
    const NpyArray &arr_r = r_it->second;
    layer->resistance.data =
        Map<const MatrixXf>(arr_r.data<float>(), arr_r.shape[0], arr_r.shape[1]);

    string vth_name = to_string(index) + ".voltage_threshold";
    auto vth_it = data.find(vth_name);
    if (vth_it == data.end()) {
      throw runtime_error("Voltage threshold array not found for module: " + to_string(index));
    }
    const NpyArray &arr_vth = vth_it->second;
    layer->voltage_threshold.data =
        Map<const MatrixXf>(arr_vth.data<float>(), arr_vth.shape[0], arr_vth.shape[1]);
  }

public:
  /**
   * @brief Saves the full model architecture and parameters to a .npz file.
   * @param model The Sequential model to save.
   * @param safe_filepath Path to the output .npz file.
   * @return True on success, false on failure.
   */
  static auto saveNetwork(const Sequential &model, const string &safe_filepath) -> bool {
    try {
      auto file_path = path(safe_filepath);
      create_directories(file_path.parent_path());

      string arch_metadata_str;
      map<string, pair<vector<size_t>, const float *>> parameters;
      size_t layer_index = 0;

      for (const auto &layer : model.layers) {
        switch (getLayerType(layer)) {

        case LayerType::Linear:
          _saveLinear(                             //
              dynamic_pointer_cast<Linear>(layer), //
              layer_index,                         //
              arch_metadata_str,                   //
              parameters                           //
          );
          break;

        case LayerType::LeakyReLU:
          _saveLeakyReLU(                             //
              dynamic_pointer_cast<LeakyReLU>(layer), //
              arch_metadata_str                       //
          );
          break;

        case LayerType::Leaky:
          _saveLeaky(                             //
              dynamic_pointer_cast<Leaky>(layer), //
              layer_index,                        //
              arch_metadata_str,                  //
              parameters                          //
          );
          break;

        case LayerType::ReLU:
          _saveReLU(arch_metadata_str);
          break;

        case LayerType::Unknown:
          cerr << "Warning: Unknown layer type at index " << layer_index
               << " encountered during serialization. It will be skipped.\n";
          break;
        }

        layer_index++;
      }

      vector<char> arch_metadata_vec(arch_metadata_str.begin(), arch_metadata_str.end());

      npz_save(                       //
          safe_filepath,              //
          "__architecture__",         //
          arch_metadata_vec.data(),   //
          {arch_metadata_vec.size()}, //
          "w"                         //
      );

      for (auto const &[name, info] : parameters) {
        npz_save(safe_filepath, name, info.second, info.first, "a");
      }

      cout << "Successfully saved network to file: " << safe_filepath << '\n';
      return true;

    } catch (const exception &e) {
      cerr << "Error saving network: " << e.what() << '\n';
      return false;
    }
  }

  /**
   * @brief Loads a model's architecture and parameters from a .npz file.
   * @param model The Sequential model to load into (will be cleared and rebuilt).
   * @param safe_filepath Path to the input .npz file.
   * @return True on success, false on failure.
   */
  static auto loadNetwork(Sequential &model, const string &safe_filepath) -> bool {
    try {
      if (!exists(safe_filepath)) {
        throw runtime_error("Network file does not exist: " + safe_filepath);
      }

      npz_t data = npz_load(safe_filepath);

      auto arch_it = data.find("__architecture__");
      if (arch_it == data.end()) {
        throw runtime_error("Architecture metadata not found in file: " + safe_filepath);
      }
      const NpyArray &arch_arr = arch_it->second;
      string arch_str(arch_arr.data<char>(), arch_arr.shape[0]);

      vector<string> arch_lines;

      size_t start = 0;
      size_t end = 0;

      while ((end = arch_str.find('\n', start)) != string::npos) {
        if (end > start) {
          arch_lines.push_back(arch_str.substr(start, end - start));
        }

        start = end + 1;
      }

      model.layers.clear();
      for (const auto &line : arch_lines) {

        vector<string> tokens;

        size_t prev = 0;
        size_t pos = 0;
        while ((pos = line.find(':', prev)) != string::npos) {
          tokens.push_back(line.substr(prev, pos - prev));
          prev = pos + 1;
        }
        tokens.push_back(line.substr(prev));

        const string &layer_type = tokens[0];

        if (layer_type == "Linear") {
          model.layers.push_back(make_shared<Linear>(stoi(tokens[1]), stoi(tokens[2])));
        } else if (layer_type == "Leaky") {
          model.layers.push_back(make_shared<Leaky>(stof(tokens[1]), stof(tokens[2]),
                                                    stof(tokens[3]), stof(tokens[4]),
                                                    tokens[5] == "1", stof(tokens[6])));
        } else if (layer_type == "LeakyReLU") {
          model.layers.push_back(make_shared<LeakyReLU>(stof(tokens[1])));
        } else if (layer_type == "ReLU") {
          model.layers.push_back(make_shared<ReLU>());
        }
      }

      size_t layer_index = 0;
      for (auto &layer : model.layers) {
        switch (getLayerType(layer)) {
        case LayerType::Linear:
          _loadLinearParams(dynamic_pointer_cast<Linear>(layer), layer_index, data);
          break;
        case LayerType::Leaky:
          _loadLeakyParams(dynamic_pointer_cast<Leaky>(layer), layer_index, data);
          break;
        case LayerType::LeakyReLU: // No params
        case LayerType::ReLU:      // No params
        case LayerType::Unknown:   // Skip
          break;
        }
        layer_index++;
      }

      cout << "Successfully loaded network from file: " << safe_filepath << "\n";
      return true;
    } catch (const exception &e) {
      cerr << "Error loading network: " << e.what() << "\n";
      return false;
    }
  }
};
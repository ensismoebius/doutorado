#pragma once

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../layers/Linear.hpp"
#include "../layers/Sequential.hpp"
#include <cnpy.h>

// NetworkSerializer class to save and load neural network weights and biases
class NetworkSerializer {
private:
  static constexpr const char *WEIGHTS_SUFFIX = "_w";
  static constexpr const char *BIAS_SUFFIX = "_b";

  /**
   * @brief Add a layer's weights and biases to the data map
   *
   * @param layer The layer to save
   * @param name Layer name for array naming
   */
  static void addLayerToData(const std::shared_ptr<Linear> &layer, const std::string &name) {
    // Add weights
    std::string weight_name = name + WEIGHTS_SUFFIX;
    std::vector<size_t> weight_shape = {static_cast<size_t>(layer->weight.data.rows()),
                                        static_cast<size_t>(layer->weight.data.cols())};

    // Add bias
    std::string bias_name = name + BIAS_SUFFIX;
    std::vector<size_t> bias_shape = {static_cast<size_t>(layer->bias.data.rows()),
                                      static_cast<size_t>(layer->bias.data.cols())};
  }

  /**
   * @brief Load a single layer's weights and biases from npz data
   *
   * @param layer The layer to load into
   * @param name Layer name for array naming
   * @param data The loaded npz data
   */
  static void loadLayer(const std::shared_ptr<Linear> &layer, const std::string &name,
                        const cnpy::npz_t &data) {
    // Load weights
    std::string weight_name = name + WEIGHTS_SUFFIX;
    auto w_it = data.find(weight_name);
    if (w_it == data.end()) {
      throw std::runtime_error("Weight array not found for layer: " + name);
    }
    const cnpy::NpyArray &arr_w = w_it->second;
    const auto *weight_data = arr_w.data<float>();
    layer->weight.data = Eigen::Map<const Eigen::MatrixXf>(weight_data, layer->weight.data.rows(),
                                                           layer->weight.data.cols());

    // Load bias
    std::string bias_name = name + BIAS_SUFFIX;
    auto b_it = data.find(bias_name);
    if (b_it == data.end()) {
      throw std::runtime_error("Bias array not found for layer: " + name);
    }
    const cnpy::NpyArray &arr_b = b_it->second;
    const auto *bias_data = arr_b.data<float>();
    layer->bias.data = Eigen::Map<const Eigen::MatrixXf>(bias_data, layer->bias.data.rows(),
                                                         layer->bias.data.cols());
  }

public:
  /**
   * @brief Save all weights and biases of a Sequential model to a single npz file
   *
   * @param model The Sequential model to save
   * @param dirpath Path to the directory to save the .npz file
   * @param layerNames Vector of names for each Linear layer
   * @return true if save was successful, false otherwise
   */
  static auto saveNetwork(const Sequential &model, const std::string &dirpath,
                          const std::vector<std::string> &layerNames) -> bool {
    try {

      bool first_layer = true;     // Flag to indicate the first layer being saved
      size_t linearLayerCount = 0; // Counter for linear layers
      const std::string safe_filepath = dirpath + "/model_weights.npz"; // Safe file path for saving

      // Create the directory if it doesn't exist
      std::filesystem::create_directories(dirpath);

      // Save all layers to a single npz file
      for (const auto &layer : model.layers) {

        // Cast the generic layer to a Linear layer
        auto linearLayer = std::dynamic_pointer_cast<Linear>(layer);

        // Check if the layer is a Linear layer
        if (linearLayer) {

          // Check if we have enough layer names
          if (linearLayerCount >= layerNames.size()) {
            throw std::runtime_error("Not enough layer names provided");
          }

          // Create weight name
          std::string weight_name = layerNames[linearLayerCount] + WEIGHTS_SUFFIX;

          // Get weight data and shape
          const float *weight_data = linearLayer->weight.data.data();
          std::vector<size_t> weight_shape = {static_cast<size_t>(linearLayer->weight.data.rows()),
                                              static_cast<size_t>(linearLayer->weight.data.cols())};

          // First array creates the file, others append to it
          cnpy::npz_save(safe_filepath, weight_name, weight_data, weight_shape,
                         first_layer ? "w" : "a");

          // Create bias name
          std::string bias_name = layerNames[linearLayerCount] + BIAS_SUFFIX;

          // Get weight data and shape
          const float *bias_data = linearLayer->bias.data.data();
          std::vector<size_t> bias_shape = {static_cast<size_t>(linearLayer->bias.data.rows()),
                                            static_cast<size_t>(linearLayer->bias.data.cols())};

          // Always append after the first array
          cnpy::npz_save(safe_filepath, bias_name, bias_data, bias_shape, "a");

          first_layer = false;
          linearLayerCount++;
        }
      }

      std::cout << "Successfully saved network to file: " << safe_filepath << "\n";
      return true;

    } catch (const std::exception &e) {
      std::cerr << "Error saving network: " << e.what() << "\n";
      return false;
    }
  }

  /**
   * @brief Load all weights and biases from a single npz file into a Sequential model
   *
   * @param model The Sequential model to load into
   * @param filepath Path to the .npz file to load
   * @param layerNames Vector of names for each Linear layer
   * @return true if load was successful, false otherwise
   */
  static auto loadNetwork(Sequential &model, const std::string & /*filepath*/,
                          const std::vector<std::string> &layerNames) -> bool {
    try {
      const std::string weights_dir = ".";
      const std::string safe_filepath = weights_dir + "/model_weights.npz";
      if (!std::filesystem::exists(safe_filepath)) {
        throw std::runtime_error("Network file does not exist: " + safe_filepath);
      }

      // Load the npz file
      cnpy::npz_t data = cnpy::npz_load(safe_filepath);
      size_t linearLayerCount = 0;

      for (const auto &layer : model.layers) {
        auto linearLayer = std::dynamic_pointer_cast<Linear>(layer);
        if (linearLayer) {
          if (linearLayerCount >= layerNames.size()) {
            throw std::runtime_error("Not enough layer names provided");
          }
          loadLayer(linearLayer, layerNames[linearLayerCount], data);
          linearLayerCount++;
        }
      }

      std::cout << "Successfully loaded network from file: " << safe_filepath << "\n";
      return true;

    } catch (const std::exception &e) {
      std::cerr << "Error loading network: " << e.what() << "\n";
      return false;
    }
  }
};
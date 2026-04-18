#ifndef NN_SAVER_NETWORKSERIALIZER_HPP
#define NN_SAVER_NETWORKSERIALIZER_HPP

#include <cnpy.h>

#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/layers/activations/LeakyReLU.hpp"
#include "nn/layers/activations/ReLU.hpp"
#include "nn/layers/base/Sequential.hpp"
#include "nn/layers/dense/Linear.hpp"
#include "nn/layers/spiking/Leaky.hpp"
#include "nn/logging/Logger.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file NetworkSerializer.hpp
 * @brief Serialize/deserialize a `Sequential` model to NumPy `.npz` via cnpy.
 *
 * What gets stored:
 * - An architecture metadata string under the key `__architecture__`.
 * - Parameter arrays keyed by layer index + suffix (e.g. `"3.weight"`, `"3.bias"`).
 *
 * Supported layers:
 * - `Linear` (weights + bias)
 * - `Leaky` (scalar 1x1 tensor params: resistance, voltage_threshold, capacitance)
 * - `ReLU`, `LeakyReLU` (no parameters)
 *
 * Limitations / caveats:
 * - This is not a general-purpose checkpoint format; it is intentionally narrow
 *   to support demos.
 * - Layer indices are positional in `model.layers`. If you insert/remove layers,
 *   old checkpoints may not load as intended.
 */

using cnpy::NpyArray;
using cnpy::npz_load;
using cnpy::npz_save;
using cnpy::npz_t;
using std::cerr;
using std::cout;
using std::dynamic_pointer_cast;
using std::exception;
using std::function;
using std::make_shared;
using std::map;
using std::pair;
using std::runtime_error;
using std::shared_ptr;
using std::stof;
using std::stoi;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;
using std::filesystem::create_directories;
using std::filesystem::exists;
using std::filesystem::path;

class NetworkSerializer
{
   public:
    static auto saveNetwork(const Sequential& model, const string& safe_filepath) -> bool;
    static auto loadNetwork(Sequential& model, const string& safe_filepath) -> bool;

   private:
    // --- Constants ---
    static constexpr const char* kWeightsSuffix = ".weight";
    static constexpr const char* kBiasSuffix = ".bias";
    static constexpr const char* kArchitectureKey = "__architecture__";

    // --- Save Handlers ---
    static void _saveLinear(const shared_ptr<Linear>& layer,
        size_t index,
        string& arch_str,
        map<string, pair<vector<size_t>, const float*>>& params);
    static void _saveLeakyReLU(const shared_ptr<LeakyReLU>& layer, string& arch_str);
    static void _saveReLU(string& arch_str);
    static void _saveLeaky(const shared_ptr<Leaky>& layer,
        size_t index,
        string& arch_str,
        map<string, pair<vector<size_t>, const float*>>& params);

    // --- Load Handlers ---
    static void _loadLinearParams(const shared_ptr<Linear>& layer, size_t index, const npz_t& data);
    static void _loadLeakyParams(const shared_ptr<Leaky>& layer, size_t index, const npz_t& data);

    // --- Helper for splitting strings ---
    static auto _split(const string& s, char delimiter) -> vector<string>
    {
        vector<string> tokens;
        string token;
        stringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter))
        {
            tokens.push_back(token);
        }
        return tokens;
    }
};

// --- Save Implementations ---

inline auto NetworkSerializer::saveNetwork(const Sequential& model, const string& safe_filepath)
    -> bool
{
    try
    {
        auto file_path = path(safe_filepath);
        create_directories(file_path.parent_path());

        string arch_metadata_str;
        map<string, pair<vector<size_t>, const float*>> parameters;
        size_t layer_index = 0;

        for (const auto& layer : model.layers)
        {
            if (auto linear = dynamic_pointer_cast<Linear>(layer))
            {
                _saveLinear(linear, layer_index, arch_metadata_str, parameters);
            }
            else if (auto leaky_relu = dynamic_pointer_cast<LeakyReLU>(layer))
            {
                _saveLeakyReLU(leaky_relu, arch_metadata_str);
            }
            else if (auto leaky = dynamic_pointer_cast<Leaky>(layer))
            {
                _saveLeaky(leaky, layer_index, arch_metadata_str, parameters);
            }
            else if (dynamic_pointer_cast<ReLU>(layer))
            {
                _saveReLU(arch_metadata_str);
            }
            else
            {
                cerr << "Warning: Unknown layer type at index " << layer_index
                     << " encountered during serialization. It will be skipped.\n";
            }
            layer_index++;
        }

        vector<char> arch_metadata_vec(arch_metadata_str.begin(), arch_metadata_str.end());
        npz_save(safe_filepath,
            kArchitectureKey,
            arch_metadata_vec.data(),
            {arch_metadata_vec.size()},
            "w");

        for (auto const& [name, info] : parameters)
        {
            npz_save(safe_filepath, name, info.second, info.first, "a");
        }

        cout << "Successfully saved network to file: " << safe_filepath << '\n';
        return true;
    }
    catch (const exception& e)
    {
        cerr << "Error saving network: " << e.what() << '\n';
        return false;
    }
}

inline void NetworkSerializer::_saveLinear(const shared_ptr<Linear>& layer,
    size_t index,
    string& arch_str,
    map<string, pair<vector<size_t>, const float*>>& params)
{
    arch_str +=
        "Linear:" + to_string(layer->in_features) + ":" + to_string(layer->out_features) + "\n";
    params[to_string(index) + kWeightsSuffix] = {
        {layer->weight.rows(), layer->weight.cols()}, layer->weight.data_ptr()};
    params[to_string(index) + kBiasSuffix] = {{layer->bias.rows()}, layer->bias.data_ptr()};
}

inline void NetworkSerializer::_saveLeakyReLU(const shared_ptr<LeakyReLU>& layer, string& arch_str)
{
    arch_str += "LeakyReLU:" + to_string(layer->alpha) + "\n";
}

inline void NetworkSerializer::_saveReLU(string& arch_str)
{
    arch_str += "ReLU\n";
}

inline void NetworkSerializer::_saveLeaky(const shared_ptr<Leaky>& layer,
    size_t index,
    string& arch_str,
    map<string, pair<vector<size_t>, const float*>>& params)
{
    arch_str += "Leaky:" + to_string(layer->time_step) + ":" +
                to_string(layer->resistance.at(0, 0)) + ":" +
                to_string(layer->capacitance.at(0, 0)) + ":" +
                to_string(layer->voltage_threshold.at(0, 0)) + ":" +
                (layer->reset_zero ? "1" : "0") + ":" + to_string(layer->reset_potential) + "\n";
    params[to_string(index) + ".resistance"] = {
        {layer->resistance.rows(), layer->resistance.cols()}, layer->resistance.data_ptr()};
    params[to_string(index) + ".voltage_threshold"] = {
        {layer->voltage_threshold.rows(), layer->voltage_threshold.cols()},
        layer->voltage_threshold.data_ptr()};
    params[to_string(index) + ".capacitance"] = {
        {layer->capacitance.rows(), layer->capacitance.cols()}, layer->capacitance.data_ptr()};
}

// --- Load Implementations ---

inline auto NetworkSerializer::loadNetwork(Sequential& model, const string& safe_filepath) -> bool
{
    try
    {
        const auto data = npz_load(safe_filepath);

        auto arch_it = data.find(kArchitectureKey);
        if (arch_it == data.end())
        {
            NN_LOG_ERROR("NetworkSerializer: architecture metadata missing in: " + safe_filepath);
            return false;
        }

        const NpyArray& arch_arr = arch_it->second;
        const char* arch_ptr = reinterpret_cast<const char*>(arch_arr.data<char>());
        size_t arch_len = static_cast<size_t>(arch_arr.shape[0]);
        std::string arch_metadata(arch_ptr, arch_ptr + arch_len);

        auto lines = _split(arch_metadata, '\n');
        size_t layer_index = 0;
        for (const auto& line : lines)
        {
            if (line.empty())
            {
                continue;
            }
            if (line.rfind("Linear:", 0) == 0)
            {
                auto parts = _split(line, ':');
                if (parts.size() < 3) throw runtime_error("Malformed Linear metadata");
                int in_f = stoi(parts[1]);
                int out_f = stoi(parts[2]);
                auto layer = make_shared<Linear>(in_f, out_f);
                _loadLinearParams(layer, layer_index, data);
                model.layers.push_back(layer);
            }
            else if (line.rfind("LeakyReLU:", 0) == 0)
            {
                auto parts = _split(line, ':');
                float alpha = 0.01F;
                if (parts.size() >= 2) alpha = stof(parts[1]);
                auto layer = make_shared<LeakyReLU>(alpha);
                model.layers.push_back(layer);
            }
            else if (line == "ReLU")
            {
                model.layers.push_back(make_shared<ReLU>());
            }
            else if (line.rfind("Leaky:", 0) == 0)
            {
                auto parts = _split(line, ':');
                if (parts.size() < 7) throw runtime_error("Malformed Leaky metadata");
                float time_step = stof(parts[1]);
                float resistance = stof(parts[2]);
                float capacitance = stof(parts[3]);
                float voltage_threshold = stof(parts[4]);
                bool reset_zero = (parts[5] == "1");
                float reset_potential = stof(parts[6]);
                auto layer = make_shared<Leaky>(time_step,
                    resistance,
                    capacitance,
                    voltage_threshold,
                    reset_zero,
                    reset_potential);
                _loadLeakyParams(layer, layer_index, data);
                model.layers.push_back(layer);
            }
            else
            {
                NN_LOG_WARN("NetworkSerializer: unknown layer metadata: '" + line + "'");
            }
            ++layer_index;
        }

        return true;
    }
    catch (const exception& e)
    {
        NN_LOG_ERROR("NetworkSerializer::loadNetwork failed: " + std::string(e.what()));
        return false;
    }
}

inline void NetworkSerializer::_loadLinearParams(
    const shared_ptr<Linear>& layer, size_t index, const npz_t& data)
{
    string weight_name = to_string(index) + kWeightsSuffix;
    auto w_it = data.find(weight_name);
    if (w_it == data.end())
    {
        throw runtime_error("Weight array not found for module: " + to_string(index));
    }
    const NpyArray& arr_w = w_it->second;
    const auto* weight_data = arr_w.data<float>();
    // Copy weights from npz array to tensor
    size_t w_idx = 0;
    for (size_t i = 0; i < layer->weight.rows(); ++i)
    {
        for (size_t j = 0; j < layer->weight.cols(); ++j)
        {
            layer->weight.at(i, j) = weight_data[w_idx++];
        }
    }

    string bias_name = to_string(index) + kBiasSuffix;
    auto b_it = data.find(bias_name);
    if (b_it == data.end())
    {
        throw runtime_error("Bias array not found for module: " + to_string(index));
    }
    const NpyArray& arr_b = b_it->second;
    const auto* bias_data = arr_b.data<float>();

    if (arr_b.shape.size() == 1)
    { // Handle 1D bias array
        for (size_t i = 0; i < static_cast<size_t>(arr_b.shape[0]); ++i)
        {
            layer->bias.at(i, 0) = bias_data[i];
        }
    }
    else
    { // Handle 2D bias array
        size_t b_idx = 0;
        for (size_t i = 0; i < layer->bias.rows(); ++i)
        {
            for (size_t j = 0; j < layer->bias.cols(); ++j)
            {
                layer->bias.at(i, j) = bias_data[b_idx++];
            }
        }
    }
}

inline void NetworkSerializer::_loadLeakyParams(
    const std::shared_ptr<Leaky>& layer, size_t index, const cnpy::npz_t& data)
{
    std::string res_name = std::to_string(index) + ".resistance";
    auto r_it = data.find(res_name);
    if (r_it == data.end())
    {
        throw std::runtime_error("Resistance array not found for module: " + std::to_string(index));
    }
    const cnpy::NpyArray& arr_r = r_it->second;
    const float* r_data = arr_r.data<float>();
    size_t r_rows = static_cast<size_t>(arr_r.shape[0]);
    size_t r_cols = static_cast<size_t>(arr_r.shape[1]);

    // Copy data element by element
    for (size_t i = 0; i < r_rows; ++i)
    {
        for (size_t j = 0; j < r_cols; ++j)
        {
            layer->resistance.at(i, j) = r_data[i * r_cols + j];
        }
    }

    std::string vth_name = std::to_string(index) + ".voltage_threshold";
    auto vth_it = data.find(vth_name);
    if (vth_it == data.end())
    {
        throw std::runtime_error(
            "Voltage threshold array not found for module: " + std::to_string(index));
    }
    const cnpy::NpyArray& arr_vth = vth_it->second;
    const float* vth_data = arr_vth.data<float>();
    size_t vth_rows = static_cast<size_t>(arr_vth.shape[0]);
    size_t vth_cols = static_cast<size_t>(arr_vth.shape[1]);

    // Copy data element by element
    for (size_t i = 0; i < vth_rows; ++i)
    {
        for (size_t j = 0; j < vth_cols; ++j)
        {
            layer->voltage_threshold.at(i, j) = vth_data[i * vth_cols + j];
        }
    }

    // Load capacitance (may be absent in older saves — fall back to current value)
    std::string cap_name = std::to_string(index) + ".capacitance";
    auto cap_it = data.find(cap_name);
    if (cap_it != data.end())
    {
        const cnpy::NpyArray& arr_c = cap_it->second;
        const float* c_data = arr_c.data<float>();
        size_t c_rows = static_cast<size_t>(arr_c.shape[0]);
        size_t c_cols = static_cast<size_t>(arr_c.shape[1]);
        for (size_t i = 0; i < c_rows; ++i)
        {
            for (size_t j = 0; j < c_cols; ++j)
            {
                layer->capacitance.at(i, j) = c_data[i * c_cols + j];
            }
        }
    }
}

#endif // NN_SAVER_NETWORKSERIALIZER_HPP

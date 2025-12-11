#pragma once

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

#include "../layers/Leaky.hpp"
#include "../layers/LeakyReLU.hpp"
#include "../layers/Linear.hpp"
#include "../layers/ReLU.hpp"
#include "../layers/Sequential.hpp"

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
    static constexpr const char* WEIGHTS_SUFFIX = ".weight";
    static constexpr const char* BIAS_SUFFIX = ".bias";
    static constexpr const char* ARCHITECTURE_KEY = "__architecture__";

    // --- Save Handlers ---
    static void _saveLinear(const shared_ptr<Linear>& layer, size_t index, string& arch_str,
                            map<string, pair<vector<size_t>, const float*>>& params);
    static void _saveLeakyReLU(const shared_ptr<LeakyReLU>& layer, string& arch_str);
    static void _saveReLU(string& arch_str);
    static void _saveLeaky(const shared_ptr<Leaky>& layer, size_t index, string& arch_str,
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
                 ARCHITECTURE_KEY,
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

inline void NetworkSerializer::_saveLinear(const shared_ptr<Linear>& layer, size_t index,
                                           string& arch_str,
                                           map<string, pair<vector<size_t>, const float*>>& params)
{
    arch_str +=
        "Linear:" + to_string(layer->in_features) + ":" + to_string(layer->out_features) + "\n";
    params[to_string(index) + WEIGHTS_SUFFIX] = {{(size_t) layer->weight.get_data_ref().rows(),
                                                  (size_t) layer->weight.get_data_ref().cols()},
                                                 layer->weight.get_data_ref().data()};
    params[to_string(index) + BIAS_SUFFIX] = {{(size_t) layer->bias.get_data_ref().rows()},
                                              layer->bias.get_data_ref().data()};
}

inline void NetworkSerializer::_saveLeakyReLU(const shared_ptr<LeakyReLU>& layer, string& arch_str)
{
    arch_str += "LeakyReLU:" + to_string(layer->alpha) + "\n";
}

inline void NetworkSerializer::_saveReLU(string& arch_str)
{
    arch_str += "ReLU\n";
}

inline void NetworkSerializer::_saveLeaky(const shared_ptr<Leaky>& layer, size_t index,
                                          string& arch_str,
                                          map<string, pair<vector<size_t>, const float*>>& params)
{
    arch_str += "Leaky:" + to_string(layer->dt) + ":" +
                to_string(layer->resistance.get_data_ref()(0, 0)) + ":" +
                to_string(layer->capacitance) + ":" +
                to_string(layer->voltage_threshold.get_data_ref()(0, 0)) + ":" +
                (layer->reset_zero ? "1" : "0") + ":" + to_string(layer->reset_potential) + "\n";
    params[to_string(index) + ".resistance"] = {{(size_t) layer->resistance.get_data_ref().rows(),
                                                 (size_t) layer->resistance.get_data_ref().cols()},
                                                layer->resistance.get_data_ref().data()};
    params[to_string(index) + ".voltage_threshold"] = {
        {(size_t) layer->voltage_threshold.get_data_ref().rows(),
         (size_t) layer->voltage_threshold.get_data_ref().cols()},
        layer->voltage_threshold.get_data_ref().data()};
}

// --- Load Implementations ---

inline auto NetworkSerializer::loadNetwork(Sequential& model, const string& safe_filepath) -> bool
{
    try
    {
        if (!exists(safe_filepath))
        {
            throw runtime_error("Network file does not exist: " + safe_filepath);
        }

        npz_t data = npz_load(safe_filepath);

        auto arch_it = data.find(ARCHITECTURE_KEY);
        if (arch_it == data.end())
        {
            throw runtime_error("Architecture metadata not found in file: " + safe_filepath);
        }
        const NpyArray& arch_arr = arch_it->second;
        string arch_str(arch_arr.data<char>(), arch_arr.shape[0]);

        using LayerFactory = function<shared_ptr<Module>(const vector<string>&)>;
        const map<string, LayerFactory> layer_factories = {
            {"Linear",
             [](const vector<string>& tokens) -> shared_ptr<Module>
             { return make_shared<Linear>(stoi(tokens[1]), stoi(tokens[2])); }},
            {"Leaky",
             [](const vector<string>& tokens) -> shared_ptr<Module>
             {
                 return make_shared<Leaky>(stof(tokens[1]),
                                           stof(tokens[2]),
                                           stof(tokens[3]),
                                           stof(tokens[4]),
                                           tokens[5] == "1",
                                           stof(tokens[6]));
             }},
            {"LeakyReLU",
             [](const vector<string>& tokens) -> shared_ptr<Module>
             { return make_shared<LeakyReLU>(stof(tokens[1])); }},
            {"ReLU",
             [](const vector<string>&) -> shared_ptr<Module> { return make_shared<ReLU>(); }}};

        model.layers.clear();
        stringstream arch_stream(arch_str);
        string line;

        while (std::getline(arch_stream, line))
        {
            if (line.empty())
            {
                continue;
            }

            vector<string> tokens = _split(line, ':');
            const string& layer_type = tokens[0];

            auto it = layer_factories.find(layer_type);
            if (it != layer_factories.end())
            {
                model.layers.push_back(it->second(tokens));
            }
            else
            {
                cerr << "Warning: Unknown layer type '" << layer_type
                     << "' in architecture string. It will be skipped.\n";
            }
        }

        size_t layer_index = 0;
        for (auto& layer : model.layers)
        {
            if (auto linear = dynamic_pointer_cast<Linear>(layer))
            {
                _loadLinearParams(linear, layer_index, data);
            }
            else if (auto leaky = dynamic_pointer_cast<Leaky>(layer))
            {
                _loadLeakyParams(leaky, layer_index, data);
            }
            // LeakyReLU and ReLU have no params to load.
            layer_index++;
        }

        cout << "Successfully loaded network from file: " << safe_filepath << "\n";
        return true;
    }
    catch (const exception& e)
    {
        cerr << "Error loading network: " << e.what() << "\n";
        return false;
    }
}

inline void NetworkSerializer::_loadLinearParams(const shared_ptr<Linear>& layer, size_t index,
                                                 const npz_t& data)
{
    string weight_name = to_string(index) + WEIGHTS_SUFFIX;
    auto w_it = data.find(weight_name);
    if (w_it == data.end())
    {
        throw runtime_error("Weight array not found for module: " + to_string(index));
    }
    const NpyArray& arr_w = w_it->second;
    layer->weight.get_data_ref() = Map<const MatrixXf>(arr_w.data<float>(),
                                                       layer->weight.get_data_ref().rows(),
                                                       layer->weight.get_data_ref().cols());

    string bias_name = to_string(index) + BIAS_SUFFIX;
    auto b_it = data.find(bias_name);
    if (b_it == data.end())
    {
        throw runtime_error("Bias array not found for module: " + to_string(index));
    }
    const NpyArray& arr_b = b_it->second;
    const auto* bias_data = arr_b.data<float>();

    if (arr_b.shape.size() == 1)
    { // Handle 1D bias array
        for (Index i = 0; i < static_cast<Index>(arr_b.shape[0]); ++i)
        {
            layer->bias.get_data_ref()(i, 0) = bias_data[i];
        }
    }
    else
    { // Handle 2D bias array
        layer->bias.get_data_ref() = Map<const MatrixXf>(
            bias_data, layer->bias.get_data_ref().rows(), layer->bias.get_data_ref().cols());
    }
}

inline void NetworkSerializer::_loadLeakyParams(const shared_ptr<Leaky>& layer, size_t index,
                                                const npz_t& data)
{
    string res_name = to_string(index) + ".resistance";
    auto r_it = data.find(res_name);
    if (r_it == data.end())
    {
        throw runtime_error("Resistance array not found for module: " + to_string(index));
    }
    const NpyArray& arr_r = r_it->second;
    layer->resistance.get_data_ref() = Map<const MatrixXf>(arr_r.data<float>(),
                                                           static_cast<Index>(arr_r.shape[0]),
                                                           static_cast<Index>(arr_r.shape[1]));

    string vth_name = to_string(index) + ".voltage_threshold";
    auto vth_it = data.find(vth_name);
    if (vth_it == data.end())
    {
        throw runtime_error("Voltage threshold array not found for module: " + to_string(index));
    }
    const NpyArray& arr_vth = vth_it->second;
    layer->voltage_threshold.get_data_ref() =
        Map<const MatrixXf>(arr_vth.data<float>(),
                            static_cast<Index>(arr_vth.shape[0]),
                            static_cast<Index>(arr_vth.shape[1]));
}

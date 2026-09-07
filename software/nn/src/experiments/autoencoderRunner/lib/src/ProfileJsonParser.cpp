/**
 * @file src/experiments/autoencoderRunner/lib/src/ProfileJsonParser.cpp
 * @brief Implementation of the hand-rolled JSON parsing helpers declared in
 *        ProfileJsonParser.hpp (extracted from ProfileLoader.cpp).
 */

#include "ProfileJsonParser.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <sstream>
#include <unordered_set>

namespace autoencoderRunner::detail
{

namespace
{
std::string normalize(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
    {
        if (std::isalnum(static_cast<unsigned char>(c)))
            out.push_back(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

auto skip_ws(const std::string& text, std::size_t pos) -> std::size_t
{
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
    return pos;
}

auto parse_quoted(const std::string& text, std::size_t start, std::string& out) -> bool
{
    if (start >= text.size() || text[start] != '"') return false;
    ++start;
    std::string result;
    for (std::size_t i = start; i < text.size(); ++i)
    {
        char c = text[i];
        if (c == '\\')
        {
            if (i + 1 < text.size())
            {
                result.push_back(text[i + 1]);
                ++i;
            }
            continue;
        }
        if (c == '"')
        {
            out = result;
            return true;
        }
        result.push_back(c);
    }
    return false;
}

auto parse_json_string_at(
    const std::string& text, std::size_t start, std::string& out, std::size_t& end_pos) -> bool
{
    if (start >= text.size() || text[start] != '"') return false;

    std::string result;
    bool escaped = false;
    for (std::size_t i = start + 1; i < text.size(); ++i)
    {
        const char c = text[i];
        if (escaped)
        {
            result.push_back(c);
            escaped = false;
            continue;
        }
        if (c == '\\')
        {
            escaped = true;
            continue;
        }
        if (c == '"')
        {
            out = std::move(result);
            end_pos = i;
            return true;
        }
        result.push_back(c);
    }

    return false;
}

auto collect_top_level_keys(const std::string& text) -> std::vector<std::string>
{
    std::vector<std::string> keys;
    int object_depth = 0;
    int array_depth = 0;
    bool in_string = false;
    bool escaped = false;
    bool expecting_key = false;

    for (std::size_t i = 0; i < text.size(); ++i)
    {
        const char c = text[i];

        if (in_string)
        {
            if (escaped)
            {
                escaped = false;
                continue;
            }
            if (c == '\\')
            {
                escaped = true;
                continue;
            }
            if (c == '"')
            {
                in_string = false;
            }
            continue;
        }

        if (c == '"')
        {
            if (object_depth == 1 && array_depth == 0 && expecting_key)
            {
                std::string key;
                std::size_t end_pos = i;
                if (!parse_json_string_at(text, i, key, end_pos))
                {
                    return keys;
                }
                keys.push_back(std::move(key));
                i = end_pos;
                expecting_key = false;
            }
            else
            {
                in_string = true;
            }
            continue;
        }

        if (c == '{')
        {
            ++object_depth;
            if (object_depth == 1 && array_depth == 0) expecting_key = true;
            continue;
        }

        if (c == '}')
        {
            if (object_depth == 1 && array_depth == 0) expecting_key = false;
            if (object_depth > 0) --object_depth;
            continue;
        }

        if (c == '[')
        {
            ++array_depth;
            continue;
        }

        if (c == ']')
        {
            if (array_depth > 0) --array_depth;
            continue;
        }

        if (object_depth == 1 && array_depth == 0 && c == ',')
        {
            expecting_key = true;
        }
    }

    return keys;
}

auto is_known_profile_key(const std::string& key) -> bool
{
    if (key.rfind("_comment", 0) == 0) return true;

    static const std::unordered_set<std::string> kKnownKeys = {
        "training_batch_size",
        "training_max_batches_per_epoch",
        "program_device",
        "dataset_subject_filter_regex",
        "dataset_root_path",
        "sampler_shuffle_samples",
        "sampler_shuffle_seed",
        "sampler_default_type",
        "sampler_weights",
        "sampler_weighted_num_samples",
        "sampler_distributed_num_replicas",
        "sampler_distributed_rank",
        "sampler_distributed_shuffle",
        "sampler_distributed_drop_last",
        "dataset_input_mode",
        "dataset_type",
        "neural_network_type",
        "neural_network_hidden_size",
        "neural_network_latent_size",
        "neural_network_depth",
        "neural_network_layer_sizes",
        "neural_network_layer",
        "neural_network_input_features",
        "neural_network_eeg_features",
        "neural_network_audio_features",
        "neural_network_architecture",
        "neural_network_branch_hidden_size",
        "neural_network_fusion_hidden_size",
        "neural_network_residual_blocks",
        "neural_network_time_step",
        "neural_network_resistance",
        "neural_network_capacitance",
        "training_optimizer_type",
        "training_learning_rate",
        "training_optimizer_momentum",
        "training_adam_beta1",
        "training_adam_beta2",
        "training_adam_epsilon",
        "training_loss_type",
        "training_epochs",
        "training_lr_plateau_enabled",
        "training_lr_plateau_factor",
        "training_lr_plateau_patience",
        "training_lr_plateau_min_delta",
        "training_normalize_inputs",
        "validation_modality_diagnostics_enabled",
        "program_prefetch_lookahead",
        "program_prefetch_ram_cap_mb",
        "window_eeg_config",
        "window_audio_config",
        "kfold_enabled",
        "kfold_n_splits",
        "kfold_shuffle",
        "kfold_seed",
        "program_opencl_profiling_enabled",
    };

    return kKnownKeys.contains(key);
}

} // namespace

auto map_dataset_type(const std::string& s, Config& cfg) -> bool
{
    auto n = normalize(s);
    if (n.find("protocol") != std::string::npos)
    {
        cfg.dataset_type = AutoencoderRunnerDatasetType::Protocol;
        return true;
    }
    if (n.find("eeg") != std::string::npos)
    {
        cfg.dataset_type = AutoencoderRunnerDatasetType::EegWindow;
        return true;
    }
    if (n.find("audio") != std::string::npos)
    {
        cfg.dataset_type = AutoencoderRunnerDatasetType::AudioWindow;
        return true;
    }
    if (n.find("fused") != std::string::npos)
    {
        cfg.dataset_type = AutoencoderRunnerDatasetType::FusedWindow;
        return true;
    }
    return false;
}

auto map_autoencoder_type(const std::string& s, Config& cfg) -> bool
{
    auto n = normalize(s);
    if (n == "protocolann")
    {
        cfg.autoencoder_type = AutoencoderRunnerAutoencoderType::ProtocolAnn;
        return true;
    }
    if (n == "protocolsnn")
    {
        cfg.autoencoder_type = AutoencoderRunnerAutoencoderType::ProtocolSnn;
        return true;
    }
    if (n == "eegwindowann")
    {
        cfg.autoencoder_type = AutoencoderRunnerAutoencoderType::EegWindowAnn;
        return true;
    }
    if (n == "eegwindowsnn")
    {
        cfg.autoencoder_type = AutoencoderRunnerAutoencoderType::EegWindowSnn;
        return true;
    }
    if (n == "audiowindowann")
    {
        cfg.autoencoder_type = AutoencoderRunnerAutoencoderType::AudioWindowAnn;
        return true;
    }
    if (n == "audiowindowsnn")
    {
        cfg.autoencoder_type = AutoencoderRunnerAutoencoderType::AudioWindowSnn;
        return true;
    }
    if (n == "fusedwindowann")
    {
        cfg.autoencoder_type = AutoencoderRunnerAutoencoderType::FusedWindowAnn;
        return true;
    }
    if (n == "fusedwindowsnn")
    {
        cfg.autoencoder_type = AutoencoderRunnerAutoencoderType::FusedWindowSnn;
        return true;
    }
    return false;
}

auto read_file(const std::filesystem::path& path, std::string& out) -> bool
{
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::ostringstream oss;
    oss << ifs.rdbuf();
    out = oss.str();
    return true;
}

auto find_key(const std::string& text, const std::string& key) -> std::size_t
{
    return text.find("\"" + key + "\"");
}

auto value_start(const std::string& text, const std::string& key, std::size_t& out_pos) -> bool
{
    const std::size_t key_pos = find_key(text, key);
    if (key_pos == std::string::npos) return false;
    std::size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos) return false;
    out_pos = skip_ws(text, colon + 1);
    return out_pos < text.size();
}

auto parse_token(const std::string& text, std::size_t start, std::string& out) -> bool
{
    if (start >= text.size()) return false;
    std::size_t end = start;
    while (end < text.size() && text[end] != ',' && text[end] != '}' && text[end] != ']') ++end;
    if (end == start) return false;
    out = text.substr(start, end - start);
    // trim
    std::size_t b = 0;
    while (b < out.size() && std::isspace(static_cast<unsigned char>(out[b]))) ++b;
    std::size_t e = out.size();
    while (e > b && std::isspace(static_cast<unsigned char>(out[e - 1]))) --e;
    out = out.substr(b, e - b);
    return !out.empty();
}

auto parse_bool(const std::string& text, const std::string& key, bool& out) -> bool
{
    std::size_t pos = 0;
    if (!value_start(text, key, pos)) return false;
    std::string token;
    if (!parse_token(text, pos, token)) return false;
    const auto n = normalize(token);
    if (n == "true")
    {
        out = true;
        return true;
    }
    if (n == "false")
    {
        out = false;
        return true;
    }
    return false;
}

auto parse_string(const std::string& text, const std::string& key, std::string& out) -> bool
{
    std::size_t pos = 0;
    if (!value_start(text, key, pos)) return false;
    return parse_quoted(text, pos, out);
}

auto parse_array_numbers(const std::string& text, const std::string& key, std::vector<double>& out)
    -> bool
{
    std::size_t pos = 0;
    if (!value_start(text, key, pos)) return false;
    if (pos >= text.size() || text[pos] != '[') return false;
    std::size_t end = text.find(']', pos);
    if (end == std::string::npos) return false;
    const std::string body = text.substr(pos + 1, end - pos - 1);

    out.clear();
    std::stringstream ss(body);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        char* p = nullptr;
        const double v = std::strtod(item.c_str(), &p);
        if (p != item.c_str()) out.push_back(v);
    }
    return true;
}

auto parse_array_ints(const std::string& text, const std::string& key, std::vector<int>& out)
    -> bool
{
    std::vector<double> vals;
    if (!parse_array_numbers(text, key, vals)) return false;
    out.clear();
    out.reserve(vals.size());
    std::transform(vals.begin(),
        vals.end(),
        std::back_inserter(out),
        [](double v) { return static_cast<int>(v); });
    return true;
}

auto parse_array_strings(
    const std::string& text, const std::string& key, std::vector<std::string>& out) -> bool
{
    std::size_t pos = 0;
    if (!value_start(text, key, pos)) return false;
    if (pos >= text.size() || text[pos] != '[') return false;

    std::size_t i = pos + 1;
    out.clear();
    while (i < text.size())
    {
        i = skip_ws(text, i);
        if (i >= text.size()) return false;
        if (text[i] == ']')
        {
            return true;
        }
        std::string value;
        if (!parse_quoted(text, i, value)) return false;
        out.push_back(std::move(value));
        i = text.find_first_of(",]", i + 1);
        if (i == std::string::npos) return false;
        if (text[i] == ']') return true;
        ++i;
    }
    return false;
}

auto parse_object(const std::string& text, const std::string& key, std::string& out) -> bool
{
    std::size_t pos = 0;
    if (!value_start(text, key, pos)) return false;
    if (pos >= text.size() || text[pos] != '{') return false;

    int depth = 0;
    for (std::size_t i = pos; i < text.size(); ++i)
    {
        if (text[i] == '{')
            ++depth;
        else if (text[i] == '}')
        {
            --depth;
            if (depth == 0)
            {
                out = text.substr(pos, i - pos + 1);
                return true;
            }
        }
    }
    return false;
}

auto validate_known_profile_keys(const std::string& text, std::string& out_error) -> bool
{
    auto top_level_keys = collect_top_level_keys(text);
    std::vector<std::string> unknown_keys;
    unknown_keys.reserve(top_level_keys.size());

    std::copy_if(top_level_keys.begin(),
        top_level_keys.end(),
        std::back_inserter(unknown_keys),
        [](const auto& key) { return !is_known_profile_key(key); });

    if (unknown_keys.empty()) return true;

    std::sort(unknown_keys.begin(), unknown_keys.end());
    unknown_keys.erase(std::unique(unknown_keys.begin(), unknown_keys.end()), unknown_keys.end());

    std::ostringstream oss;
    oss << "unknown profile key(s): ";
    for (std::size_t i = 0; i < unknown_keys.size(); ++i)
    {
        if (i > 0) oss << ", ";
        oss << unknown_keys[i];
    }
    out_error = oss.str();
    return false;
}

} // namespace autoencoderRunner::detail

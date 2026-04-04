/**
 * @file src/experiments/03/lib/src/ProfileLoader.cpp
 * @brief Implementation for Profileloader.
 *

 */

// ProfileLoader.cpp
#include "ProfileLoader.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

namespace experiment03
{
static std::string normalize(const std::string& s)
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

static bool map_dataset_type(const std::string& s, Config& cfg)
{
    auto n = normalize(s);
    if (n.find("protocol") != std::string::npos)
    {
        cfg.dataset_type = Experiment03DatasetType::Protocol;
        return true;
    }
    if (n.find("eeg") != std::string::npos)
    {
        cfg.dataset_type = Experiment03DatasetType::EegWindow;
        return true;
    }
    if (n.find("audio") != std::string::npos)
    {
        cfg.dataset_type = Experiment03DatasetType::AudioWindow;
        return true;
    }
    if (n.find("fused") != std::string::npos)
    {
        cfg.dataset_type = Experiment03DatasetType::FusedWindow;
        return true;
    }
    return false;
}

static bool map_autoencoder_type(const std::string& s, Config& cfg)
{
    auto n = normalize(s);
    bool is_snn = n.find("snn") != std::string::npos || n.find("spiking") != std::string::npos;
    if (n.find("protocol") != std::string::npos)
    {
        cfg.autoencoder_type = is_snn ? Experiment03AutoencoderType::ProtocolSnn
                                      : Experiment03AutoencoderType::ProtocolAnn;
        return true;
    }
    if (n.find("eeg") != std::string::npos)
    {
        cfg.autoencoder_type = is_snn ? Experiment03AutoencoderType::EegWindowSnn
                                      : Experiment03AutoencoderType::EegWindowAnn;
        return true;
    }
    if (n.find("audio") != std::string::npos)
    {
        cfg.autoencoder_type = is_snn ? Experiment03AutoencoderType::AudioWindowSnn
                                      : Experiment03AutoencoderType::AudioWindowAnn;
        return true;
    }
    if (n.find("fused") != std::string::npos)
    {
        cfg.autoencoder_type = is_snn ? Experiment03AutoencoderType::FusedWindowSnn
                                      : Experiment03AutoencoderType::FusedWindowAnn;
        return true;
    }
    return false;
}

static auto read_file(const std::filesystem::path& path, std::string& out) -> bool
{
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::ostringstream oss;
    oss << ifs.rdbuf();
    out = oss.str();
    return true;
}

static auto find_key(const std::string& text, const std::string& key) -> std::size_t
{
    return text.find("\"" + key + "\"");
}

static auto skip_ws(const std::string& text, std::size_t pos) -> std::size_t
{
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
    return pos;
}

static auto value_start(const std::string& text, const std::string& key, std::size_t& out_pos)
    -> bool
{
    const std::size_t key_pos = find_key(text, key);
    if (key_pos == std::string::npos) return false;
    std::size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos) return false;
    out_pos = skip_ws(text, colon + 1);
    return out_pos < text.size();
}

static auto parse_quoted(const std::string& text, std::size_t start, std::string& out) -> bool
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

static auto parse_token(const std::string& text, std::size_t start, std::string& out) -> bool
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

template <typename T>
static auto parse_number(const std::string& text, const std::string& key, T& out) -> bool
{
    std::size_t pos = 0;
    if (!value_start(text, key, pos)) return false;
    std::string token;
    if (!parse_token(text, pos, token)) return false;
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str()) return false;
    out = static_cast<T>(value);
    return true;
}

static auto parse_bool(const std::string& text, const std::string& key, bool& out) -> bool
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

static auto parse_string(const std::string& text, const std::string& key, std::string& out) -> bool
{
    std::size_t pos = 0;
    if (!value_start(text, key, pos)) return false;
    return parse_quoted(text, pos, out);
}

static auto parse_array_numbers(
    const std::string& text, const std::string& key, std::vector<double>& out) -> bool
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

static auto parse_array_ints(const std::string& text, const std::string& key, std::vector<int>& out)
    -> bool
{
    std::vector<double> vals;
    if (!parse_array_numbers(text, key, vals)) return false;
    out.clear();
    out.reserve(vals.size());
    for (double v : vals)
    {
        out.push_back(static_cast<int>(v));
    }
    return true;
}

static auto parse_object(const std::string& text, const std::string& key, std::string& out) -> bool
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

static auto parse_json_string_at(
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

static auto collect_top_level_keys(const std::string& text) -> std::vector<std::string>
{
    std::vector<std::string> keys;
    int depth = 0;
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
            if (depth == 1 && expecting_key)
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
            ++depth;
            if (depth == 1) expecting_key = true;
            continue;
        }

        if (c == '}')
        {
            if (depth == 1) expecting_key = false;
            if (depth > 0) --depth;
            continue;
        }

        if (depth == 1 && c == ',')
        {
            expecting_key = true;
        }
    }

    return keys;
}

static auto is_known_profile_key(const std::string& key) -> bool
{
    if (key.rfind("_comment", 0) == 0) return true;

    static constexpr std::array<std::string_view, 43> kKnownKeys = {
        "training_batch_size",
        "training_max_batches_per_epoch",
        "device",
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
        "autoencoder_type",
        "autoencoder_hidden_size",
        "autoencoder_latent_size",
        "autoencoder_depth",
        "autoencoder_layer_sizes",
        "layer_sizes",
        "hidden_layer_sizes",
        "autoencoder_input_features",
        "autoencoder_eeg_features",
        "autoencoder_audio_features",
        "input_features",
        "eeg_features",
        "audio_features",
        "layers",
        "autoencoder_architecture",
        "autoencoder_branch_hidden_size",
        "autoencoder_fusion_hidden_size",
        "autoencoder_residual_blocks",
        "autoencoder_time_step",
        "autoencoder_resistance",
        "autoencoder_capacitance",
        "training_learning_rate",
        "training_epochs",
        "prefetch_lookahead",
        "prefetch_ram_cap_mb",
        "window_eeg_config",
        "window_audio_config",
    };

    return std::find(kKnownKeys.begin(), kKnownKeys.end(), key) != kKnownKeys.end();
}

static auto validate_known_profile_keys(const std::string& text, std::string& out_error) -> bool
{
    auto top_level_keys = collect_top_level_keys(text);
    std::vector<std::string> unknown_keys;
    unknown_keys.reserve(top_level_keys.size());

    for (const auto& key : top_level_keys)
    {
        if (!is_known_profile_key(key)) unknown_keys.push_back(key);
    }

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

auto load_profile_to_config(
    const std::string& profile_name, Config& out_config, std::string& out_error) -> bool
{
    namespace fs = std::filesystem;

    const fs::path source_profiles_dir =
        fs::path(__FILE__).parent_path().parent_path().parent_path() / "profiles";

    const fs::path raw_profile_path(profile_name);
    const bool looks_like_path = raw_profile_path.is_absolute() ||
                                 profile_name.find('/') != std::string::npos ||
                                 profile_name.find('\\') != std::string::npos;
    const bool has_json_extension = raw_profile_path.extension() == ".json";

    std::vector<fs::path> candidates = {raw_profile_path,
        source_profiles_dir / (profile_name + ".json"),
        fs::path("src/experiments/03/profiles") / (profile_name + ".json"),
        fs::current_path() / (profile_name + ".json"),
        fs::path("profiles") / (profile_name + ".json")};

    if (!has_json_extension)
    {
        candidates.insert(candidates.begin() + 1, raw_profile_path.string() + ".json");
    }

    if (looks_like_path && has_json_extension)
    {
        // Also try path exactly as provided before profile-name fallbacks.
        candidates.insert(candidates.begin(), raw_profile_path);
    }

    fs::path selected;
    for (auto& p : candidates)
    {
        if (p.empty()) continue;
        if (fs::exists(p))
        {
            selected = p;
            break;
        }
    }

    if (selected.empty())
    {
        out_error = "profile not found in known locations for '" + profile_name + "'";
        return false;
    }

    std::string text;
    if (!read_file(selected, text))
    {
        out_error = "failed to read profile: " + selected.string();
        return false;
    }

    if (!validate_known_profile_keys(text, out_error))
    {
        return false;
    }

    parse_number(text, "training_batch_size", out_config.training_batch_size);
    parse_number(text, "training_max_batches_per_epoch", out_config.training_max_batches_per_epoch);
    parse_string(text, "device", out_config.device);
    parse_string(text, "dataset_subject_filter_regex", out_config.dataset_subject_filter_regex);
    parse_string(text, "dataset_root_path", out_config.dataset_root_path);
    parse_bool(text, "sampler_shuffle_samples", out_config.sampler_shuffle_samples);
    if (unsigned int sampler_shuffle_seed = 0U;
        parse_number(text, "sampler_shuffle_seed", sampler_shuffle_seed))
    {
        out_config.sampler_shuffle_seed = sampler_shuffle_seed;
    }
    parse_string(text, "sampler_default_type", out_config.sampler_default_type);

    parse_array_numbers(text, "sampler_weights", out_config.sampler_weights);
    if (out_config.sampler_weights.empty() &&
        find_key(text, "sampler_weights") != std::string::npos)
    {
        // Preserve explicit empty list in profile file.
        out_config.sampler_weights.clear();
    }

    if (std::size_t weighted_num_samples = 0;
        parse_number(text, "sampler_weighted_num_samples", weighted_num_samples))
    {
        out_config.sampler_weighted_num_samples = weighted_num_samples;
    }

    parse_number(
        text, "sampler_distributed_num_replicas", out_config.sampler_distributed_num_replicas);
    parse_number(text, "sampler_distributed_rank", out_config.sampler_distributed_rank);
    parse_bool(text, "sampler_distributed_shuffle", out_config.sampler_distributed_shuffle);
    parse_bool(text, "sampler_distributed_drop_last", out_config.sampler_distributed_drop_last);

    if (int dataset_input_mode = 0; parse_number(text, "dataset_input_mode", dataset_input_mode))
    {
        out_config.dataset_input_mode = static_cast<Protocol101117InputMode>(dataset_input_mode);
    }

    std::string str_value;
    if (parse_string(text, "dataset_type", str_value)) map_dataset_type(str_value, out_config);
    if (parse_string(text, "autoencoder_type", str_value))
        map_autoencoder_type(str_value, out_config);

    parse_number(text, "autoencoder_hidden_size", out_config.autoencoder_hidden_size);
    parse_number(text, "autoencoder_latent_size", out_config.autoencoder_latent_size);
    parse_number(text, "autoencoder_depth", out_config.autoencoder_depth);
    parse_array_ints(text, "autoencoder_layer_sizes", out_config.autoencoder_layer_sizes);
    parse_array_ints(text, "layer_sizes", out_config.autoencoder_layer_sizes);
    parse_array_ints(text, "hidden_layer_sizes", out_config.autoencoder_layer_sizes);
    parse_number(text, "autoencoder_input_features", out_config.autoencoder_input_features);
    parse_number(text, "autoencoder_eeg_features", out_config.autoencoder_eeg_features);
    parse_number(text, "autoencoder_audio_features", out_config.autoencoder_audio_features);

    // Backward/short aliases for profile files.
    parse_number(text, "input_features", out_config.autoencoder_input_features);
    parse_number(text, "eeg_features", out_config.autoencoder_eeg_features);
    parse_number(text, "audio_features", out_config.autoencoder_audio_features);
    parse_number(text, "layers", out_config.autoencoder_depth);

    if (int architecture = 0; parse_number(text, "autoencoder_architecture", architecture))
    {
        out_config.autoencoder_architecture = static_cast<AutoencoderArchitecture>(architecture);
    }

    parse_number(text, "autoencoder_branch_hidden_size", out_config.autoencoder_branch_hidden_size);
    parse_number(text, "autoencoder_fusion_hidden_size", out_config.autoencoder_fusion_hidden_size);
    parse_number(text, "autoencoder_residual_blocks", out_config.autoencoder_residual_blocks);
    parse_number(text, "autoencoder_time_step", out_config.autoencoder_time_step);
    parse_number(text, "autoencoder_resistance", out_config.autoencoder_resistance);
    parse_number(text, "autoencoder_capacitance", out_config.autoencoder_capacitance);

    parse_number(text, "training_learning_rate", out_config.training_learning_rate);
    parse_number(text, "training_epochs", out_config.training_epochs);
    parse_number(text, "prefetch_lookahead", out_config.prefetch_lookahead);
    parse_number(text, "prefetch_ram_cap_mb", out_config.prefetch_ram_cap_mb);

    std::string eeg_object;
    if (parse_object(text, "window_eeg_config", eeg_object))
    {
        parse_number(eeg_object, "window_size", out_config.window_eeg_config.window_size);
        parse_number(eeg_object, "overlap", out_config.window_eeg_config.overlap);
        parse_number(eeg_object, "sample_rate", out_config.window_eeg_config.sample_rate);
    }

    std::string audio_object;
    if (parse_object(text, "window_audio_config", audio_object))
    {
        parse_number(audio_object, "window_size", out_config.window_audio_config.window_size);
        parse_number(audio_object, "overlap", out_config.window_audio_config.overlap);
        parse_number(audio_object, "sample_rate", out_config.window_audio_config.sample_rate);
    }

    out_error.clear();
    return true;
}

} // namespace experiment03

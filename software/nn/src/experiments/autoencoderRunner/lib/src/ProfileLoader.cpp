/**
 * @file src/experiments/autoencoderRunner/lib/src/ProfileLoader.cpp
 * @brief Implementation for Profileloader.
 *

 */

// ProfileLoader.cpp
#include "ProfileLoader.hpp"

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "ProfileJsonParser.hpp"

using nn::models::autoencoder::AutoencoderArchitecture;

namespace autoencoderRunner
{

using namespace detail;

static auto parse_neural_network_layers(
    const std::string& text, Config& out_config, std::string& out_error) -> bool
{
    std::vector<std::string> entries;
    if (!parse_array_strings(text, "neural_network_layer", entries)) return true;

    out_config.autoencoder_encoder_layer_spec.clear();
    out_config.autoencoder_decoder_layer_spec.clear();
    out_config.autoencoder_branch_encoder_layer_spec.clear();
    out_config.autoencoder_branch_decoder_layer_spec.clear();
    out_config.autoencoder_fusion_encoder_layer_spec.clear();
    out_config.autoencoder_fusion_decoder_layer_spec.clear();

    for (const auto& entry : entries)
    {
        const auto separator = entry.find(':');
        if (separator == std::string::npos || separator == 0 || separator + 1 >= entry.size())
        {
            out_error = "neural_network_layer entry must use 'section:layer_spec': " + entry;
            return false;
        }

        const auto section = entry.substr(0, separator);
        const auto layer_spec = entry.substr(separator + 1);

        if (section == "encoder")
        {
            out_config.autoencoder_encoder_layer_spec.push_back(layer_spec);
            continue;
        }
        if (section == "decoder")
        {
            out_config.autoencoder_decoder_layer_spec.push_back(layer_spec);
            continue;
        }
        if (section == "branch_encoder")
        {
            out_config.autoencoder_branch_encoder_layer_spec.push_back(layer_spec);
            continue;
        }
        if (section == "branch_decoder")
        {
            out_config.autoencoder_branch_decoder_layer_spec.push_back(layer_spec);
            continue;
        }
        if (section == "fusion_encoder")
        {
            out_config.autoencoder_fusion_encoder_layer_spec.push_back(layer_spec);
            continue;
        }
        if (section == "fusion_decoder")
        {
            out_config.autoencoder_fusion_decoder_layer_spec.push_back(layer_spec);
            continue;
        }

        out_error = "unsupported neural_network_layer section: " + section;
        return false;
    }

    return true;
}

/// Where a profile might live, in the order the CLI has always searched.
///
/// The chain is deliberate: `--profile article-lstm-ae` has to resolve from
/// the build directory, from the source tree, and from a results directory,
/// while an absolute path or an explicit `.json` has to win over all of
/// them. An empty return means nothing matched -- the caller turns that
/// into the message naming what was asked for, because "not found" has to
/// say WHICH profile was not found.
auto locate_profile(const std::string& profile_name) -> std::filesystem::path
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
        fs::path("src/experiments/autoencoderRunner/profiles") / (profile_name + ".json"),
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

    return selected;
}

/// Dataset location/selection and every sampler knob.
void apply_dataset_and_sampler(const std::string& text, Config& out_config)
{
    parse_number(text, "training_batch_size", out_config.training_batch_size);
    parse_number(text, "training_max_batches_per_epoch", out_config.training_max_batches_per_epoch);
    parse_string(text, "program_device", out_config.device);
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
}

/// The model: which autoencoder, how wide, how deep, and the SNN membrane
/// constants.
///
/// The only failure here is an unknown `neural_network_type`, and it has to
/// stay a failure: ignoring an unrecognized name would leave the DEFAULT
/// model in place, and the run would train the wrong architecture while
/// reporting a perfectly normal loss.
auto apply_model(const std::string& text, Config& out_config, std::string& out_error) -> bool
{
    std::string str_value;
    if (parse_string(text, "neural_network_type", str_value) &&
        !map_autoencoder_type(str_value, out_config))
    {
        out_error = "unsupported neural_network_type: " + str_value;
        return false;
    }

    parse_number(text, "neural_network_hidden_size", out_config.autoencoder_hidden_size);
    parse_number(text, "neural_network_latent_size", out_config.autoencoder_latent_size);
    parse_number(text, "neural_network_depth", out_config.autoencoder_depth);
    parse_array_ints(text, "neural_network_layer_sizes", out_config.autoencoder_layer_sizes);
    if (!parse_neural_network_layers(text, out_config, out_error))
    {
        return false;
    }
    parse_number(text, "neural_network_input_features", out_config.autoencoder_input_features);
    parse_number(text, "neural_network_eeg_features", out_config.autoencoder_eeg_features);
    parse_number(text, "neural_network_audio_features", out_config.autoencoder_audio_features);
    if (int architecture = 0; parse_number(text, "neural_network_architecture", architecture))
    {
        out_config.autoencoder_architecture = static_cast<AutoencoderArchitecture>(architecture);
    }

    parse_number(
        text, "neural_network_branch_hidden_size", out_config.autoencoder_branch_hidden_size);
    parse_number(
        text, "neural_network_fusion_hidden_size", out_config.autoencoder_fusion_hidden_size);
    parse_number(text, "neural_network_residual_blocks", out_config.autoencoder_residual_blocks);
    parse_number(text, "neural_network_time_step", out_config.autoencoder_time_step);
    parse_number(text, "neural_network_resistance", out_config.autoencoder_resistance);
    parse_number(text, "neural_network_capacitance", out_config.autoencoder_capacitance);

    return true;
}

/// Optimizer, loss, epochs, LR schedule and input normalization.
void apply_training(const std::string& text, Config& out_config)
{
    parse_string(text, "training_optimizer_type", out_config.training_optimizer_type);
    parse_number(text, "training_learning_rate", out_config.training_learning_rate);
    parse_number(text, "training_optimizer_momentum", out_config.training_optimizer_momentum);
    parse_number(text, "training_adam_beta1", out_config.training_optimizer_adam_beta1);
    parse_number(text, "training_adam_beta2", out_config.training_optimizer_adam_beta2);
    parse_number(text, "training_adam_epsilon", out_config.training_optimizer_adam_epsilon);
    parse_string(text, "training_loss_type", out_config.training_loss_type);
    parse_number(text, "training_epochs", out_config.training_epochs);
    parse_bool(text, "training_lr_plateau_enabled", out_config.training_lr_plateau_enabled);
    parse_number(text, "training_lr_plateau_factor", out_config.training_lr_plateau_factor);
    parse_number(text, "training_lr_plateau_patience", out_config.training_lr_plateau_patience);
    parse_number(text, "training_lr_plateau_min_delta", out_config.training_lr_plateau_min_delta);
    parse_bool(text, "training_normalize_inputs", out_config.training_normalize_inputs);
    parse_bool(text,
        "validation_modality_diagnostics_enabled",
        out_config.validation_modality_diagnostics_enabled);
}

/// Prefetch, k-fold, test split, and the OpenCL profiling switch.
void apply_program_and_kfold(const std::string& text, Config& out_config)
{
    parse_number(text, "program_prefetch_lookahead", out_config.prefetch_lookahead);
    parse_number(text, "program_prefetch_ram_cap_mb", out_config.prefetch_ram_cap_mb);
    parse_bool(text, "kfold_enabled", out_config.kfold_enabled);
    parse_number(text, "kfold_n_splits", out_config.kfold_n_splits);
    parse_bool(text, "kfold_shuffle", out_config.kfold_shuffle);
    if (unsigned int kfold_seed = 0U; parse_number(text, "kfold_seed", kfold_seed))
    {
        out_config.kfold_seed = kfold_seed;
    }
    parse_number(text, "test_split", out_config.test_split);
    parse_bool(text, "program_opencl_profiling_enabled", out_config.opencl_profiling_enabled);
}

/// Per-modality windowing, which is nested in the profile rather than flat.
void apply_windowing(const std::string& text, Config& out_config)
{
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
}

auto load_profile_to_config(
    const std::string& profile_name, Config& out_config, std::string& out_error) -> bool
{
    const std::filesystem::path selected = locate_profile(profile_name);
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

    apply_dataset_and_sampler(text, out_config);
    if (!apply_model(text, out_config, out_error))
    {
        return false;
    }
    apply_training(text, out_config);
    apply_program_and_kfold(text, out_config);
    apply_windowing(text, out_config);

    out_error.clear();
    return true;
}

} // namespace autoencoderRunner

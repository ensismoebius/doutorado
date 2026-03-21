/**
 * @file src/experiments/03/lib/src/cli.cpp
 * @brief CLI parsing and option handling for Experiment03.
 *
 * Uses CLI11 to declare and parse command-line options for the experiment
 * binary. This file adapts values into the `Config` structure used by
 * `Experiment03`.
 */

#include "../include/cli.hpp"

#include <cstdlib>
#include <stdexcept>

#include "../include/ProfileLoader.hpp"
#include "nn/dataLoaders/10.1117/codec/InputModeCodec.hpp"
#include "nn/dataLoaders/SamplerOptionResolution.hpp"

namespace
{
auto has_help_flag(int argc, char* argv[]) -> bool
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";
        if (arg == "-h" || arg == "--help" || arg == "--help-all")
        {
            return true;
        }
    }

    return false;
}

auto parse_profile_name_from_argv(int argc, char* argv[], const std::string& fallback)
    -> std::string
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";
        if (arg == "--profile" && i + 1 < argc && argv[i + 1])
        {
            return argv[i + 1];
        }

        const std::string prefix = "--profile=";
        if (arg.rfind(prefix, 0) == 0)
        {
            return arg.substr(prefix.size());
        }
    }

    return fallback;
}

auto datasetTypeToToken(Experiment03DatasetType dataset_type) -> std::string
{
    switch (dataset_type)
    {
        case Experiment03DatasetType::Protocol:
            return "protocol";
        case Experiment03DatasetType::EegWindow:
            return "eeg-window";
        case Experiment03DatasetType::AudioWindow:
            return "audio-window";
        case Experiment03DatasetType::FusedWindow:
            return "fused-window";
    }

    return "protocol";
}

auto parseDatasetTypeToken(const std::string& token) -> Experiment03DatasetType
{
    if (token == "protocol") return Experiment03DatasetType::Protocol;
    if (token == "eeg-window") return Experiment03DatasetType::EegWindow;
    if (token == "audio-window") return Experiment03DatasetType::AudioWindow;
    if (token == "fused-window") return Experiment03DatasetType::FusedWindow;

    throw std::invalid_argument("Unsupported dataset type token: " + token);
}

auto autoencoderTypeToToken(Experiment03AutoencoderType autoencoder_type) -> std::string
{
    switch (autoencoder_type)
    {
        case Experiment03AutoencoderType::ProtocolAnn:
            return "protocol-ann";
        case Experiment03AutoencoderType::EegWindowAnn:
            return "eeg-window-ann";
        case Experiment03AutoencoderType::AudioWindowAnn:
            return "audio-window-ann";
        case Experiment03AutoencoderType::FusedWindowAnn:
            return "fused-window-ann";
        case Experiment03AutoencoderType::ProtocolSnn:
            return "protocol-snn";
        case Experiment03AutoencoderType::EegWindowSnn:
            return "eeg-window-snn";
        case Experiment03AutoencoderType::AudioWindowSnn:
            return "audio-window-snn";
        case Experiment03AutoencoderType::FusedWindowSnn:
            return "fused-window-snn";
    }

    return "fused-window-ann";
}

auto parseAutoencoderTypeToken(const std::string& token) -> Experiment03AutoencoderType
{
    if (token == "protocol-ann") return Experiment03AutoencoderType::ProtocolAnn;
    if (token == "eeg-window-ann") return Experiment03AutoencoderType::EegWindowAnn;
    if (token == "audio-window-ann") return Experiment03AutoencoderType::AudioWindowAnn;
    if (token == "fused-window-ann") return Experiment03AutoencoderType::FusedWindowAnn;
    if (token == "protocol-snn") return Experiment03AutoencoderType::ProtocolSnn;
    if (token == "eeg-window-snn") return Experiment03AutoencoderType::EegWindowSnn;
    if (token == "audio-window-snn") return Experiment03AutoencoderType::AudioWindowSnn;
    if (token == "fused-window-snn") return Experiment03AutoencoderType::FusedWindowSnn;

    throw std::invalid_argument("Unsupported autoencoder type token: " + token);
}

auto architectureToToken(AutoencoderArchitecture architecture) -> std::string
{
    switch (architecture)
    {
        case AutoencoderArchitecture::Auto:
            return "auto";
        case AutoencoderArchitecture::ResidualDense:
            return "residual-dense";
        case AutoencoderArchitecture::DualBranchFusion:
            return "dual-branch-fusion";
    }

    return "auto";
}

auto parseArchitectureToken(const std::string& token) -> AutoencoderArchitecture
{
    if (token == "auto") return AutoencoderArchitecture::Auto;
    if (token == "residual-dense") return AutoencoderArchitecture::ResidualDense;
    if (token == "dual-branch-fusion") return AutoencoderArchitecture::DualBranchFusion;

    throw std::invalid_argument("Unsupported autoencoder architecture token: " + token);
}
} // namespace

auto parseCliParams(int argc, char* argv[], const Config& default_config) -> Config
{
    Config config = default_config;
    std::string profile = parse_profile_name_from_argv(argc, argv, default_config.profile_name);

    // Seed defaults from profile before registering CLI options so explicit
    // command line flags still win over profile values.
    if (!has_help_flag(argc, argv))
    {
        std::string profile_error;
        if (!experiment03::load_profile_to_config(profile, config, profile_error))
        {
            throw std::runtime_error("Failed to load profile '" + profile + "': " + profile_error);
        }
    }

    config.profile_name = profile;
    std::string input_mode_token = protocol101117InputModeToToken(config.input_mode);
    std::string dataset_type_token = datasetTypeToToken(config.dataset_type);
    std::string autoencoder_type_token = autoencoderTypeToToken(config.autoencoder_type);
    std::string architecture_token = architectureToToken(config.autoencoder_architecture);

    App app("PyTorch-style loader pipeline for 10.1117 EEG+Audio dataset.");

    app.add_option("--profile", profile, "Configuration profile name (JSON file stem)")
        ->default_val(config.profile_name);

    app.add_option("--dataset-root", config.dataset_root, "Path containing subjects dir")
        ->expected(1)
        ->check(CLI::ExistingDirectory)
        ->default_val(default_config.dataset_root);

    app.add_option(                                                                     //
           "--subject",                                                                 //
           config.subject_filter_regex,                                                 //
           "Subject regex filter pattern (regex should contain a group for subject id)" //
           )
        ->expected(1)
        ->default_val(default_config.subject_filter_regex);

    app.add_option("--batch-size", config.batch_size, "Mini-batch size")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.batch_size);

    app.add_option(
           "--max-batches", config.max_batches_per_epoch, "Max batches to iterate in this demo")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.max_batches_per_epoch);

    const auto input_mode_tokens = supportedProtocol101117InputModeTokens();
    app.add_option(          //
           "--input-mode",   //
           input_mode_token, //
           "Dataset input mode (protocol only): concatenated|eeg-only|audio-only")
        ->check(CLI::IsMember(input_mode_tokens, CLI::ignore_case))
        ->default_val(protocol101117InputModeToToken(default_config.input_mode));

    app.add_option(          //
           "--dataset-type", //
           dataset_type_token,
           "Dataset variant: protocol|eeg-window|audio-window|fused-window")
        ->check(CLI::IsMember(
            {"protocol", "eeg-window", "audio-window", "fused-window"}, CLI::ignore_case))
        ->default_val(datasetTypeToToken(default_config.dataset_type));

    app.add_option(         //
           "--autoencoder", //
           autoencoder_type_token,
           "Autoencoder variant: "
           "protocol-ann|eeg-window-ann|audio-window-ann|fused-window-ann|protocol-snn|eeg-window-"
           "snn|audio-window-snn|fused-window-snn")
        ->check(CLI::IsMember({"protocol-ann",
                                  "eeg-window-ann",
                                  "audio-window-ann",
                                  "fused-window-ann",
                                  "protocol-snn",
                                  "eeg-window-snn",
                                  "audio-window-snn",
                                  "fused-window-snn"},
            CLI::ignore_case))
        ->default_val(autoencoderTypeToToken(default_config.autoencoder_type));

    app.add_option(
           "--ae-hidden-size", config.autoencoder_hidden_size, "Autoencoder hidden layer width")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.autoencoder_hidden_size);

    app.add_option("--ae-latent-size",
           config.autoencoder_latent_size,
           "Autoencoder latent (bottleneck) dimension")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(config.autoencoder_latent_size);

    app.add_option("--ae-depth",
           config.autoencoder_depth,
           "Number of hidden layers in encoder/decoder (1–N)")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(config.autoencoder_depth);

    app.add_option("--ae-layer-sizes",
           config.autoencoder_layer_sizes,
           "Explicit hidden-layer widths (comma-separated). Overrides --ae-depth/--ae-hidden-size "
           "tapering")
        ->delimiter(',')
        ->check(CLI::PositiveNumber)
        ->default_str(config.autoencoder_layer_sizes.empty() ? "(auto)" : "(set)");

    app.add_option("--ae-input-features",
           config.autoencoder_input_features,
           "Input feature count override (0 = infer from dataset batch)")
        ->expected(1)
        ->check(CLI::NonNegativeNumber)
        ->default_val(config.autoencoder_input_features);

    app.add_option("--ae-eeg-features",
           config.autoencoder_eeg_features,
           "EEG feature count override for dual-branch models (0 = infer)")
        ->expected(1)
        ->check(CLI::NonNegativeNumber)
        ->default_val(config.autoencoder_eeg_features);

    app.add_option("--ae-audio-features",
           config.autoencoder_audio_features,
           "Audio feature count override for dual-branch models (0 = infer)")
        ->expected(1)
        ->check(CLI::NonNegativeNumber)
        ->default_val(config.autoencoder_audio_features);

    app.add_option("--ae-architecture",
           architecture_token,
           "Autoencoder design family: auto|residual-dense|dual-branch-fusion")
        ->check(CLI::IsMember({"auto", "residual-dense", "dual-branch-fusion"}, CLI::ignore_case))
        ->default_val(architectureToToken(config.autoencoder_architecture));

    app.add_option("--ae-branch-hidden-size",
           config.autoencoder_branch_hidden_size,
           "Multimodal branch projection width (0 = infer from hidden/latent sizes)")
        ->expected(1)
        ->check(CLI::NonNegativeNumber)
        ->default_val(config.autoencoder_branch_hidden_size);

    app.add_option("--ae-fusion-hidden-size",
           config.autoencoder_fusion_hidden_size,
           "Shared multimodal fusion width (0 = infer from hidden/latent sizes)")
        ->expected(1)
        ->check(CLI::NonNegativeNumber)
        ->default_val(config.autoencoder_fusion_hidden_size);

    app.add_option("--ae-residual-blocks",
           config.autoencoder_residual_blocks,
           "Residual blocks per dense stage for redesigned ANN builders")
        ->expected(1)
        ->check(CLI::NonNegativeNumber)
        ->default_val(config.autoencoder_residual_blocks);

    app.add_option("--ae-time-step",
           config.autoencoder_time_step,
           "SNN neuron time step dt (also used as leaky beta scale)")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(config.autoencoder_time_step);

    app.add_option(
           "--ae-resistance", config.autoencoder_resistance, "SNN neuron membrane resistance R")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(config.autoencoder_resistance);

    app.add_option(
           "--ae-capacitance", config.autoencoder_capacitance, "SNN neuron membrane capacitance C")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(config.autoencoder_capacitance);

    app.add_option("--lr", config.training_learning_rate, "Adam optimizer learning rate")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.training_learning_rate);

    app.add_option("--epochs", config.training_epochs, "Number of training epochs")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.training_epochs);

    app.add_option("--eeg-window-size",
           config.eeg_window_config.window_size,
           "EEG window size (samples) for eeg-window/fused-window datasets")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.eeg_window_config.window_size);

    app.add_option("--eeg-overlap",
           config.eeg_window_config.overlap,
           "EEG overlap in [0,1) for eeg-window/fused-window datasets")
        ->expected(1)
        ->check(CLI::Range(0.0, 0.9999))
        ->default_val(default_config.eeg_window_config.overlap);

    app.add_option("--audio-window-size",
           config.audio_window_config.window_size,
           "Audio window size (samples) for audio-window/fused-window datasets")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.audio_window_config.window_size);

    app.add_option("--audio-overlap",
           config.audio_window_config.overlap,
           "Audio overlap in [0,1) for audio-window/fused-window datasets")
        ->expected(1)
        ->check(CLI::Range(0.0, 0.9999))
        ->default_val(default_config.audio_window_config.overlap);

    app.add_option(
           "--lookahead", config.prefetch_lookahead, "Number of batches to prefetch in background")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.prefetch_lookahead);

    app.add_option("--seed",
           config.shuffle_seed,
           "Deterministic seed for shuffling (ignored if --no-shuffle)")
        ->expected(1)
        ->check(CLI::NonNegativeNumber)
        ->default_val(default_config.shuffle_seed.value_or(0U));

    app.add_flag(                     //
           "--shuffle,!--no-shuffle", //
           config.shuffle_samples,    //
           "Shuffle samples before batching")
        ->default_val(default_config.shuffle_samples);

    auto* sampler_type_option = app.add_option(                        //
        "--sampler-type",                                              //
        config.default_sampler_type,                                   //
        "Default sampler type: sequential|random|weighted|distributed" //
    );
    sampler_type_option->check(CLI::IsMember(                //
        {"sequential", "random", "weighted", "distributed"}, //
        CLI::ignore_case));

    // Empty string means "auto" (legacy shuffle/no-shuffle behavior).
    // Use default_str (display-only) so the IsMember validator is not applied to the default.
    sampler_type_option->default_str(default_config.default_sampler_type.empty()
                                         ? "(auto)"
                                         : default_config.default_sampler_type);

    app.add_option( //
           "--sampler-weights",
           config.sampler_weights,
           "Comma-separated weights for weighted sampler (e.g. 0.1,0.2,0.7)")
        ->delimiter(',')
        ->default_str("(none)");

    app.add_option( //
           "--weighted-num-samples",
           config.weighted_sampler_num_samples,
           "Number of sampled indices per epoch for weighted sampler")
        ->check(CLI::PositiveNumber)
        ->default_str("(none)");

    app.add_option( //
           "--distributed-num-replicas",
           config.distributed_sampler_num_replicas,
           "Total number of distributed replicas")
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.distributed_sampler_num_replicas);

    app.add_option( //
           "--distributed-rank",
           config.distributed_sampler_rank,
           "Current distributed rank")
        ->check(CLI::NonNegativeNumber)
        ->default_val(default_config.distributed_sampler_rank);

    app.add_flag( //
           "--distributed-shuffle,!--distributed-no-shuffle",
           config.distributed_sampler_shuffle,
           "Shuffle globally before distributed partition")
        ->default_val(default_config.distributed_sampler_shuffle);

    app.add_flag( //
           "--distributed-drop-last,!--distributed-no-drop-last",
           config.distributed_sampler_drop_last,
           "Drop tail to make dataset divisible by replicas")
        ->default_val(default_config.distributed_sampler_drop_last);

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
        std::exit(app.exit(e));
    }

    config.profile_name = profile;

    config.default_sampler_type = normalizeSamplerTypeToken(config.default_sampler_type);

    config.input_mode = parseProtocol101117InputModeToken(input_mode_token);
    dataset_type_token = CLI::detail::to_lower(dataset_type_token);
    config.dataset_type = parseDatasetTypeToken(dataset_type_token);
    autoencoder_type_token = CLI::detail::to_lower(autoencoder_type_token);
    config.autoencoder_type = parseAutoencoderTypeToken(autoencoder_type_token);
    architecture_token = CLI::detail::to_lower(architecture_token);
    config.autoencoder_architecture = parseArchitectureToken(architecture_token);

    config.resolved_sampler_options = resolveDefaultSamplerOptions(SamplerOptionSelection{
        .sampler_type = config.default_sampler_type,
        .shuffle = config.shuffle_samples,
        .seed = config.shuffle_seed,
        .weights = config.sampler_weights,
        .weighted_num_samples = config.weighted_sampler_num_samples,
        .distributed_num_replicas = config.distributed_sampler_num_replicas,
        .distributed_rank = config.distributed_sampler_rank,
        .distributed_shuffle = config.distributed_sampler_shuffle,
        .distributed_drop_last = config.distributed_sampler_drop_last,
    });
    return config;
}
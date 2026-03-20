#include "../include/cli.hpp"

#include <stdexcept>

#include "nn/dataLoaders/10.1117/codec/InputModeCodec.hpp"
#include "nn/dataLoaders/SamplerOptionResolution.hpp"

namespace
{
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
} // namespace

auto parseCliParams(int argc, char* argv[], const Config& default_config) -> Config
{
    Config config = default_config;
    std::string input_mode_token = protocol101117InputModeToToken(default_config.input_mode);
    std::string dataset_type_token = datasetTypeToToken(default_config.dataset_type);
    std::string autoencoder_type_token = autoencoderTypeToToken(default_config.autoencoder_type);

    App app("PyTorch-style loader pipeline for 10.1117 EEG+Audio dataset.");

    app.add_option("--dataset-root", config.dataset_root, "Path containing subjects dir")
        ->expected(1)
        ->check(CLI::ExistingDirectory)
        ->default_val(default_config.dataset_root);

    app.add_option(                                                                     //
           "--subject",                                                                 //
           config.subject_regex_pattern,                                                //
           "Subject regex filter pattern (regex should contain a group for subject id)" //
           )
        ->expected(1)
        ->default_val(default_config.subject_regex_pattern);

    app.add_option("--batch-size", config.batch_size, "Mini-batch size")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.batch_size);

    app.add_option("--max-batches", config.max_batches, "Max batches to iterate in this demo")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.max_batches);

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

    app.add_option("--ae-hidden-size", config.ae_hidden_size, "Autoencoder hidden layer width")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.ae_hidden_size);

    app.add_option(
           "--ae-latent-size", config.ae_latent_size, "Autoencoder latent (bottleneck) dimension")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.ae_latent_size);

    app.add_option(
           "--ae-depth", config.ae_depth, "Number of hidden layers in encoder/decoder (1–N)")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.ae_depth);

    app.add_option("--ae-time-step",
           config.ae_time_step,
           "SNN neuron time step dt (also used as leaky beta scale)")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.ae_time_step);

    app.add_option("--ae-resistance", config.ae_resistance, "SNN neuron membrane resistance R")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.ae_resistance);

    app.add_option("--ae-capacitance", config.ae_capacitance, "SNN neuron membrane capacitance C")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.ae_capacitance);

    app.add_option("--lr", config.learning_rate, "Adam optimizer learning rate")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.learning_rate);

    app.add_option("--epochs", config.epochs, "Number of training epochs")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.epochs);

    app.add_option("--eeg-window-size",
           config.eeg_window_spec.window_size,
           "EEG window size (samples) for eeg-window/fused-window datasets")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.eeg_window_spec.window_size);

    app.add_option("--eeg-overlap",
           config.eeg_window_spec.overlap,
           "EEG overlap in [0,1) for eeg-window/fused-window datasets")
        ->expected(1)
        ->check(CLI::Range(0.0, 0.9999))
        ->default_val(default_config.eeg_window_spec.overlap);

    app.add_option("--audio-window-size",
           config.audio_window_spec.window_size,
           "Audio window size (samples) for audio-window/fused-window datasets")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.audio_window_spec.window_size);

    app.add_option("--audio-overlap",
           config.audio_window_spec.overlap,
           "Audio overlap in [0,1) for audio-window/fused-window datasets")
        ->expected(1)
        ->check(CLI::Range(0.0, 0.9999))
        ->default_val(default_config.audio_window_spec.overlap);

    app.add_option("--lookahead", config.lookahead, "Number of batches to prefetch in background")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.lookahead);

    app.add_option(
           "--seed", config.seed, "Deterministic seed for shuffling (ignored if --no-shuffle)")
        ->expected(1)
        ->check(CLI::NonNegativeNumber)
        ->default_val(default_config.seed.value_or(0U));

    app.add_flag(                            //
           "--shuffle,!--no-shuffle",        //
           config.shuffle,                   //
           "Shuffle samples before batching" //
           )
        ->default_val(default_config.shuffle);

    auto* sampler_type_option = app.add_option(                        //
        "--sampler-type",                                              //
        config.sampler_type,                                           //
        "Default sampler type: sequential|random|weighted|distributed" //
    );
    sampler_type_option->check(CLI::IsMember(                //
        {"sequential", "random", "weighted", "distributed"}, //
        CLI::ignore_case));

    // Empty string means "auto" (legacy shuffle/no-shuffle behavior).
    // Use default_str (display-only) so the IsMember validator is not applied to the default.
    sampler_type_option->default_str(
        default_config.sampler_type.empty() ? "(auto)" : default_config.sampler_type);

    app.add_option( //
           "--sampler-weights",
           config.sampler_weights,
           "Comma-separated weights for weighted sampler (e.g. 0.1,0.2,0.7)")
        ->delimiter(',')
        ->default_str("(none)");

    app.add_option( //
           "--weighted-num-samples",
           config.weighted_num_samples,
           "Number of sampled indices per epoch for weighted sampler")
        ->check(CLI::PositiveNumber)
        ->default_str("(none)");

    app.add_option( //
           "--distributed-num-replicas",
           config.distributed_num_replicas,
           "Total number of distributed replicas")
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.distributed_num_replicas);

    app.add_option( //
           "--distributed-rank",
           config.distributed_rank,
           "Current distributed rank")
        ->check(CLI::NonNegativeNumber)
        ->default_val(default_config.distributed_rank);

    app.add_flag( //
           "--distributed-shuffle,!--distributed-no-shuffle",
           config.distributed_shuffle,
           "Shuffle globally before distributed partition")
        ->default_val(default_config.distributed_shuffle);

    app.add_flag( //
           "--distributed-drop-last,!--distributed-no-drop-last",
           config.distributed_drop_last,
           "Drop tail to make dataset divisible by replicas")
        ->default_val(default_config.distributed_drop_last);

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
        app.exit(e);
    }

    config.sampler_type = normalizeSamplerTypeToken(config.sampler_type);

    config.input_mode = parseProtocol101117InputModeToken(input_mode_token);
    dataset_type_token = CLI::detail::to_lower(dataset_type_token);
    config.dataset_type = parseDatasetTypeToken(dataset_type_token);
    autoencoder_type_token = CLI::detail::to_lower(autoencoder_type_token);
    config.autoencoder_type = parseAutoencoderTypeToken(autoencoder_type_token);

    config.sampler_options = resolveDefaultSamplerOptions(SamplerOptionSelection{
        .sampler_type = config.sampler_type,
        .shuffle = config.shuffle,
        .seed = config.seed,
        .weights = config.sampler_weights,
        .weighted_num_samples = config.weighted_num_samples,
        .distributed_num_replicas = config.distributed_num_replicas,
        .distributed_rank = config.distributed_rank,
        .distributed_shuffle = config.distributed_shuffle,
        .distributed_drop_last = config.distributed_drop_last,
    });
    return config;
}
/**
 * @file src/experiments/03/experiment03.cpp
 * @brief Lightweight launcher for Experiment03 CLI.
 *
 * This translation unit implements the small main() wrapper used to parse CLI
 * parameters and invoke the `Experiment03` driver. It is intentionally thin;
 * the experiment implementation lives under `lib/src/` and the runtime
 * configuration is declared in `lib/include/experiment03.hpp`.
 */
// FIXME - I want to run all autoencoders and compare its
// results, test shards and make sure autoencoders are workign properly
#include "lib/include/experiment03.hpp"

#include <cstddef>
#include <iostream>
#include <memory>

#include "lib/include/cli.hpp"
#include "nn/dataLoaders/10.1117/protocol/Protocol101117Dataset.hpp"
#include "nn/logging/StreamRedirector.hpp"

using std::cerr;
using std::cout;
using std::exception;
using std::make_shared;
using std::size_t;

const Config default_config{
    .device = "opencl",

    // Dataset discovery defaults.
    .dataset_subject_filter_regex = "^S(\\d+)$",
    .dataset_root_path =
        "/home/ensismoebius/Documentos"
        "/UNESP/doutorado/databases/"
        "BaseDeDatosHablaImaginada",

    // Training throughput controls.
    .training_batch_size = 15,
    .training_max_batches_per_epoch = 1500,

    // Sampling behavior. Empty sampler type keeps the legacy shuffle/no-shuffle path.
    .sampler_shuffle_samples = true,
    .sampler_shuffle_seed = 42U,
    .sampler_default_type = "",
    .sampler_weights = {},
    .sampler_weighted_num_samples = std::nullopt,
    .sampler_distributed_num_replicas = 10,
    .sampler_distributed_rank = 0,
    .sampler_distributed_shuffle = true,
    .sampler_distributed_drop_last = false,

    // Dataset/model pairing defaults.
    .dataset_input_mode = Protocol101117InputMode::Concatenated,
    .dataset_type = Experiment03DatasetType::FusedWindow,

    // Autoencoder architecture hyperparameters.
    .autoencoder_type = Experiment03AutoencoderType::FusedWindowAnn,
    .autoencoder_hidden_size = 64,
    .autoencoder_latent_size = 32,
    .autoencoder_depth = 2,
    .autoencoder_layer_sizes = {},
    .autoencoder_input_features = 0,
    .autoencoder_eeg_features = 0,
    .autoencoder_audio_features = 0,
    .autoencoder_architecture = AutoencoderArchitecture::Auto,
    .autoencoder_branch_hidden_size = 0,
    .autoencoder_fusion_hidden_size = 0,
    .autoencoder_residual_blocks = 1,
    .autoencoder_time_step = 1.0F,
    .autoencoder_resistance = 1.0F,
    .autoencoder_capacitance = 1.0F,

    // Training hyperparameters.
    .training_learning_rate = 1e-4f,
    .training_epochs = 2,

    // Window specs used by windowing datasets.
    .window_eeg_config = {.window_size = 256, .overlap = 0.5F, .sample_rate = 1024},
    .window_audio_config = {.window_size = 11025, .overlap = 0.5F, .sample_rate = 44100},

    // Background input pipeline controls.
    .prefetch_lookahead = 20,

    .prefetch_ram_cap_mb = 1000,
    .opencl_profiling_enabled = false,
};

auto main(int argc, char* argv[]) -> int
{
    Config config = parseCliParams(argc, argv, default_config);

    // Redirect stdout/stderr only after CLI parsing so --help and CLI validation
    // messages remain visible on the real console.
    nn::logging::StreamRedirector redirect(true, true);
    Experiment03 experiment(config);
    return experiment.run();
}

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
    .subject_filter_regex = "^S(\\d+)$",
    .dataset_root =
        "/home/ensismoebius/Documentos"
        "/UNESP/doutorado/databases/"
        "BaseDeDatosHablaImaginada",
    .batch_size = 10,
    .max_batches_per_epoch = 10,
    .shuffle_samples = true,
    .shuffle_seed = 42U,
    .default_sampler_type = "",
    .sampler_weights = {},
    .weighted_sampler_num_samples = std::nullopt,
    .distributed_sampler_num_replicas = 4,
    .distributed_sampler_rank = 0,
    .distributed_sampler_shuffle = true,
    .distributed_sampler_drop_last = false,
    .input_mode = Protocol101117InputMode::Concatenated,
    .dataset_type = Experiment03DatasetType::FusedWindow,
    .autoencoder_type = Experiment03AutoencoderType::FusedWindowAnn,
    .prefetch_lookahead = 5,
    .prefetch_ram_cap_mb = 0,
    .use_sqlite = true,
};

auto main(int argc, char* argv[]) -> int
{
    // Redirect stdout/stderr into the Logger so logs are shown above progress.
    nn::logging::StreamRedirector redirect(true, true);
    Config config = parseCliParams(argc, argv, default_config);
    Experiment03 experiment(config);
    return experiment.run();
}

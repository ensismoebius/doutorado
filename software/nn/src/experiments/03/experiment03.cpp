// FIXME - STOPPED HERE - This experiment is a bit of a "kitchen sink" for testing the new
// BatchPrefetcher and related utilities. It should be refactored into more focused experiments that
// test specific components in isolation, with clear assertions and expected outcomes. The current
// version is more of an integration test that demonstrates the full pipeline from dataset discovery
// to model inference, but it lacks specific checks and may be too complex for debugging individual
// issues.

#include "lib/include/experiment03.hpp"

#include <cstddef>
#include <iostream>
#include <memory>

#include "lib/include/cli.hpp"
#include "nn/dataLoaders/10.1117/Protocol101117Dataset.hpp"

using std::cerr;
using std::cout;
using std::exception;
using std::make_shared;
using std::size_t;

const Config default_config{
    .subject_regex_pattern = "^S(\\d+)$",
    .dataset_root =
        "/home/ensismoebius/Documentos"
        "/UNESP/doutorado/databases/"
        "BaseDeDatosHablaImaginada",
    .batch_size = 5,
    .max_batches = 10,
    .shuffle = true,
    .seed = 42U,
    .sampler_type = "",
    .sampler_weights = {},
    .weighted_num_samples = std::nullopt,
    .distributed_num_replicas = 1,
    .distributed_rank = 0,
    .distributed_shuffle = true,
    .distributed_drop_last = false,
    .input_mode = Protocol101117InputMode::Concatenated,
    .lookahead = 5,
};

auto main(int argc, char* argv[]) -> int
{
    Config config = parseCliParams(argc, argv, default_config);
    Experiment03 experiment(config);
    return experiment.run();
}

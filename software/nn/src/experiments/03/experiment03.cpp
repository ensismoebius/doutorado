// FIXME - STOPPED HERE - I have to use the Datasets recently implemented in the nn module, but I
// need to adapt them to the current code structure. I also need to implement the training loop and
// the model architecture for this experiment.
#include "lib/include/experiment03.hpp"

#include <cstddef>
#include <iostream>
#include <memory>

#include "lib/include/cli.hpp"
#include "nn/dataLoaders/10.1117/protocol/Protocol101117Dataset.hpp"

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
    .dataset_type = Experiment03DatasetType::FusedWindow,
    .autoencoder_type = Experiment03AutoencoderType::FusedWindowAnn,
    .lookahead = 5,
};

auto main(int argc, char* argv[]) -> int
{
    Config config = parseCliParams(argc, argv, default_config);
    Experiment03 experiment(config);
    return experiment.run();
}

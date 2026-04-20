/**
 * @file src/experiments/03/lib/include/experiment04.hpp
 * @brief Integrated Experiment04 runner hosted under the Experiment03 module.
 *
 * This API keeps the LSTM autoencoder experiment available after folding the
 * old `src/experiments/04` subtree into `src/experiments/03/lib`.
 */

#pragma once

namespace lstm_autoencoder_experiment
{
auto should_run_from_cli(int argc, char* argv[]) -> bool;
}

class LstmAutoencoderExperiment
{
   public:
    auto run(int argc, char* argv[]) -> int;
};
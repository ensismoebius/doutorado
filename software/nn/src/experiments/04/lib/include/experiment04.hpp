/**
 * @file src/experiments/04/lib/include/experiment04.hpp
 * @brief Integrated Experiment04 runner public interface.
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
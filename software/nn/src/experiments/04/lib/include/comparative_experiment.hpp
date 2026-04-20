#pragma once

/**
 * @file comparative_experiment.hpp
 * @brief Deterministic comparative experiment runner for SNN-AE vs LSTM-AE.
 */

class ComparativeAutoencoderExperiment
{
   public:
    auto run(int argc, char* argv[]) -> int;
};

namespace comparative_autoencoder_experiment
{
auto should_run_from_cli(int argc, char* argv[]) -> bool;
}

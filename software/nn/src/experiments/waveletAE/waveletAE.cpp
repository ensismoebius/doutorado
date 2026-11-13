#include <iostream>
#include <string>

#include "WaveletAEConfig.hpp"
/**
 * @file waveletAE.cpp
 * @brief Thin entry point for Experiment 02.
 */

#include "WaveletAEPipeline.hpp"

auto main(int argc, char const* const* argv) -> int
{
    // No StreamRedirector here: it suppresses the Logger's console output until a progress
    // printer drains it, and this binary has none, so every message (errors included) vanished.
    try
    {
        if (argc != 2)
        {
            std::cerr << "usage: " << argv[0] << " <spec.json>  (see spec.example.json)\n";
            return 2;
        }
        const std::string spec_path = argv[1];

        ExperimentConfig config = load_experiment_config(spec_path);
        run_wavelet_baseline_experiment(config);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
#include <filesystem>
#include <iostream>
#include <string>

#include "WaveletAEConfig.hpp"
/**
 * @file waveletAE.cpp
 * @brief Thin entry point for Experiment 02.
 */

#include "WaveletAEPipeline.hpp"
#include "logging/StreamRedirector.hpp"

auto main(int argc, char const* const* argv) -> int
{
    nn::logging::StreamRedirector redirect(true, true);
    try
    {
        std::string spec_path = "../src/experiments/waveletAE/spec.json";
        if (!std::filesystem::exists(spec_path))
        {
            spec_path = "src/experiments/waveletAE/spec.json";
        }
        if (argc > 1)
        {
            spec_path = argv[1];
        }

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
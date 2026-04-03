/**
 * @file src/experiments/02/Experiment02Pipeline.hpp
 * @brief Experiment02pipeline.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_EXPERIMENTS_02_EXPERIMENT02PIPELINE_HPP
#define NN_EXPERIMENTS_02_EXPERIMENT02PIPELINE_HPP

#include "Experiment02Config.hpp"

auto run_wavelet_baseline_experiment(const ExperimentConfig& config) -> void;

#endif // NN_EXPERIMENTS_02_EXPERIMENT02PIPELINE_HPP

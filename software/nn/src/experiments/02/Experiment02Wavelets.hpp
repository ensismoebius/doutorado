/**
 * @file src/experiments/02/Experiment02Wavelets.hpp
 * @brief Experiment02wavelets.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_EXPERIMENTS_02_EXPERIMENT02WAVELETS_HPP
#define NN_EXPERIMENTS_02_EXPERIMENT02WAVELETS_HPP

#include <string>
#include <vector>

#include "nn/wavelet/waveletOperations.h"

auto get_wavelet_coeffs(
    const std::string& wavelet_name, const std::vector<double>& signal, int max_level)
    -> wavelets::WaveletTransformResults;

#endif // NN_EXPERIMENTS_02_EXPERIMENT02WAVELETS_HPP

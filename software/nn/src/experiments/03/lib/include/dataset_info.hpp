/**
 * @file src/experiments/03/lib/include/dataset_info.hpp
 * @brief Utilities for printing dataset summaries used by Experiment03.
 */

#pragma once

#include <string>

class Protocol101117Dataset;

/**
 * Print a short summary of the dataset including subjects and per-subject sample counts.
 *
 * This helper is intended for interactive demos; it uses a lightweight
 * estimation strategy for cross-modal sample counts to avoid expensive I/O.
 */
void printDatasetSummary(const Protocol101117Dataset& dataset, const std::string& dataset_root);

/**
 * @file src/experiments/03/lib/include/dataset_info.hpp
 * @brief Utilities for printing dataset summaries used by Experiment03.
 */

#pragma once

#include <string>

#include "nn/dataLoaders/IDatasetPrinter.hpp"

// Printers have been moved to dedicated headers under the new datasets/ layout.
#include "nn/dataLoaders/10.1117/datasets/raw/Dataset101117Printer.hpp"
#include "nn/dataLoaders/10.1117/datasets/windowed/WindowingDatasetPrinter.hpp"

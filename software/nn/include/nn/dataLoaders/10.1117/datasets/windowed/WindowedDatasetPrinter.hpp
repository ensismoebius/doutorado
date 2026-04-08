/**
 * @file include/nn/dataLoaders/10.1117/datasets/windowed/WindowedDatasetPrinter.hpp
 * @brief Simple implementation of IDatasetPrinter for windowed datasets.
 */

#ifndef NN_DATALOADERS_10_1117_DATASETS_WINDOWED_WINDOWEDDATASETPRINTER_HPP
#define NN_DATALOADERS_10_1117_DATASETS_WINDOWED_WINDOWEDDATASETPRINTER_HPP

#include <iostream>

#include "nn/dataLoaders/IDatasetPrinter.hpp"

/**
 * Basic printer for windowed datasets.
 * Prints only the dataset size and provides a simple summary.
 */
class WindowedDatasetPrinter : public IDatasetPrinter
{
   public:
    /**
     * Print a windowed dataset with basic information.
     * @param dataset The dataset to print.
     */
    void print_generic(const Dataset& dataset) override
    {
        std::cout << "Dataset initialized with " << dataset.size() << " total samples." << '\n';
    }
};

#endif // NN_DATALOADERS_10_1117_DATASETS_WINDOWED_WINDOWEDDATASETPRINTER_HPP

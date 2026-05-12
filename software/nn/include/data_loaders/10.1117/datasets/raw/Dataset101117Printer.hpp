/**
 * @file include/nn/dataLoaders/10.1117/datasets/raw/Dataset101117Printer.hpp
 * @brief Declaration of `Dataset101117Printer` moved into datasets/raw layout.
 */

#pragma once

#include <string>

#include "data_loaders/interfaces/IDatasetPrinter.hpp"

class Dataset;
class Dataset101117;

class Dataset101117Printer : public IDatasetPrinter
{
   public:
    explicit Dataset101117Printer(const std::string& dataset_root);

    // Default generic printing fallback
    void print_generic(const Dataset& dataset) override;

    // Protocol-specific printing
    void print_protocol101117(const Dataset101117& dataset);

   private:
    std::string dataset_root_;
};

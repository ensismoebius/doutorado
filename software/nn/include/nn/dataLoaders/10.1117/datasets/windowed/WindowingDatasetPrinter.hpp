/**
 * @file include/nn/dataLoaders/10.1117/datasets/windowed/WindowingDatasetPrinter.hpp
 * @brief Declaration of `WindowingDatasetPrinter` moved into datasets/windowed layout.
 */

#pragma once

#include <string>

#include "nn/dataLoaders/IDatasetPrinter.hpp"

class Dataset;
class AudioWindowDataset;
class EEGWindowDataset;
class FusedWindowDataset;

class WindowingDatasetPrinter : public IDatasetPrinter
{
   public:
    explicit WindowingDatasetPrinter(const std::string& context);

    void print_generic(const Dataset& dataset) override;

    void print_audio_window(const AudioWindowDataset& ds);
    void print_eeg_window(const EEGWindowDataset& ds);
    void print_fused_window(const FusedWindowDataset& ds);

   private:
    std::string context_;
};

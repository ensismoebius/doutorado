/**
 * @file lfcc_pipeline_utils.h
 * @brief LFCC pipeline helpers used by the demo/experiment code.
 *
 * These declarations expose a small, testable surface:
 * - load and process audio from a `.mat` file
 * - per-subject orchestration helpers
 */

#ifndef EXPERIMENT01_UTILS_HPP
#define EXPERIMENT01_UTILS_HPP

#include "nn/tensor/Tensor.hpp" // For Tensor
#include "nn/wave/audioTypes.h" // Include the new audio types header

// Declaration for loadAndProcessAudio, now defined in Experiment01_utils.cpp
auto loadAndProcessAudio(const std::string& audioFilePath,
                         const LoadingAndProcessingParameters& loading_params)
    -> std::vector<nn::Tensor>;

void processSubject(const SubjectInfo& subject);

#endif // EXPERIMENT01_UTILS_HPP

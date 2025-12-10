#ifndef EXPERIMENT01_UTILS_HPP
#define EXPERIMENT01_UTILS_HPP

#include "core/tensor/Tensor.hpp" // For Tensor
#include "core/wave/audioTypes.h" // Include the new audio types header

// Declaration for loadAndProcessAudio, now defined in Experiment01_utils.cpp
auto loadAndProcessAudio(const std::string& audioFilePath,
                         const LoadingAndProcessingParameters& loading_params)
    -> std::vector<nn::Tensor>;

void processSubject(const SubjectInfo& subject);

#endif // EXPERIMENT01_UTILS_HPP

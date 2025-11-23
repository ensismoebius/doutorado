#ifndef EXPERIMENT01_UTILS_HPP
#define EXPERIMENT01_UTILS_HPP

#include <cstddef> // For size_t
#include <string>
#include <vector> // For std::vector

#include "core/tensor/Tensor.hpp" // For Tensor
#include "core/wave/audioFeatureExtraction.h" // Include the new header
#include "core/wave/audioTypes.h" // Include the new audio types header

void processSubject(const SubjectInfo& subject);

#endif // EXPERIMENT01_UTILS_HPP

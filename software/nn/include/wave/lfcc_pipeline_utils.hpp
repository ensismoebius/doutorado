/**
 * @file lfcc_pipeline_utils.hpp
 * @brief LFCC pipeline helpers exposed from core wave utilities.
 */

#ifndef NN_WAVE_LFCC_PIPELINE_UTILS_H
#define NN_WAVE_LFCC_PIPELINE_UTILS_H

#include <string>
#include <vector>

#include "tensor/Tensor.hpp"
#include "wave/audioTypes.hpp"

auto load_and_process_audio(const std::string& audio_file_path,
    const LoadingAndProcessingParameters& loading_params) -> std::vector<nn::Tensor>;

void process_subject(const SubjectInfo& subject);

#endif // NN_WAVE_LFCC_PIPELINE_UTILS_H
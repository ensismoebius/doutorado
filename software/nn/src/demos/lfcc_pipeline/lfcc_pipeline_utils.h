/**
 * @file lfcc_pipeline_utils.h
 * @brief LFCC pipeline helpers used by the demo/experiment code.
 *
 * These declarations expose a small, testable surface:
 * - load and process audio from a `.mat` file
 * - per-subject orchestration helpers
 */

#ifndef LFCC_PIPELINE_UTILS_HPP
#define LFCC_PIPELINE_UTILS_HPP

#include "nn/tensor/Tensor.hpp" // For Tensor
#include "nn/wave/audioTypes.h" // Include the new audio types header

auto load_and_process_audio(const std::string& audio_file_path,
                            const LoadingAndProcessingParameters& loading_params)
    -> std::vector<nn::Tensor>;

void process_subject(const SubjectInfo& subject);

// Backward-compatible wrappers for legacy callers.
inline auto loadAndProcessAudio(const std::string& audioFilePath,
                                const LoadingAndProcessingParameters& loading_params)
    -> std::vector<nn::Tensor>
{
    return load_and_process_audio(audioFilePath, loading_params);
}

inline void processSubject(const SubjectInfo& subject)
{
    process_subject(subject);
}

#endif // LFCC_PIPELINE_UTILS_HPP

#include "AudioLoader.h"

#include <matio.h>

#include <memory>
#include <stdexcept>

namespace nn::dataLoaders
{

auto loadAudioFromMat(const std::string& filePath, size_t rowIndex)
    -> std::tuple<Eigen::VectorXf, int>
{
    // Define a lambda for closing the MAT file
    auto matFileDeleter = [](mat_t* mat) { Mat_Close(mat); };

    // Open the MAT file using a smart pointer with the lambda deleter
    std::unique_ptr<mat_t, decltype(matFileDeleter)> matFile(
        Mat_Open(filePath.c_str(), MAT_ACC_RDONLY), // Open the MAT file
        matFileDeleter                              // Use the lambda as the deleter
    );

    if (!matFile)
    {
        throw std::runtime_error("Failed to open MAT file: " + filePath);
    }

    // Read the audio variable, using Mat_VarFree directly as the deleter.
    std::unique_ptr<matvar_t, decltype(&Mat_VarFree)> audioVariable(
        Mat_VarRead(matFile.get(), AUDIO_VARIABLE_NAME), // Read the audio variable
        &Mat_VarFree                                     // Use Mat_VarFree as the deleter
    );

    if (!audioVariable)
    {
        throw std::runtime_error("Failed to read audio variable from MAT file");
    }

    // Verify dimensions (M_rows x MATRIX_COLUMNS)
    if (audioVariable->rank != 2 || audioVariable->dims[1] != MATRIX_COLUMNS)
    {
        throw std::runtime_error("Invalid matrix dimensions. Expected Mx176402");
    }

    // Verify data type is double
    if (audioVariable->class_type != MAT_C_DOUBLE)
    {
        throw std::runtime_error("Invalid matrix data type. Expected double.");
    }

    // Verify rowIndex is valid
    if (rowIndex >= audioVariable->dims[0])
    {
        throw std::runtime_error("Row index out of bounds");
    }

    // Get data pointer
    auto* rawDataPtr = static_cast<double*>(audioVariable->data);
    if (rawDataPtr == nullptr)
    {
        throw std::runtime_error("Failed to access data");
    }

    // Create Eigen vector for the audio samples
    Eigen::VectorXf audioSamples(AUDIO_SAMPLES_COUNT);

    // Copy audio samples
    for (int i = 0; i < AUDIO_SAMPLES_COUNT; ++i)
    {
        audioSamples(i) = static_cast<float>(rawDataPtr[(i * audioVariable->dims[0]) + rowIndex]);
    }

    // Get the EEG index
    int eegIndex =
        static_cast<int>(rawDataPtr[(EEG_INDEX_COLUMN * audioVariable->dims[0]) + rowIndex]);

    // Cleanup is handled automatically by unique_ptr destructors

    return {std::move(audioSamples), eegIndex};
}

} // namespace nn::dataLoaders

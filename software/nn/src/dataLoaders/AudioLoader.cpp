#include "AudioLoader.h"

#include <matio.h>

#include <stdexcept>

namespace nn::dataLoaders
{

auto loadAudioFromMat(const std::string& filePath, size_t rowIndex)
    -> std::tuple<Eigen::VectorXf, int>
{
    // Open the MAT file
    mat_t* matfp = Mat_Open(filePath.c_str(), MAT_ACC_RDONLY);
    if (matfp == nullptr)
    {
        throw std::runtime_error("Failed to open MAT file: " + filePath);
    }

    // Read the audio variable specifically
    matvar_t* matvar = Mat_VarRead(matfp, AUDIO_VARIABLE_NAME);
    if (matvar == nullptr)
    {
        Mat_Close(matfp);
        throw std::runtime_error("Failed to read audio variable from MAT file");
    }

    // Verify dimensions (M_rows x MATRIX_COLUMNS)
    if (matvar->rank != 2 || matvar->dims[1] != MATRIX_COLUMNS)
    {
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        throw std::runtime_error("Invalid matrix dimensions. Expected Mx176402");
    }

    // Verify data type is double
    if (matvar->class_type != MAT_C_DOUBLE)
    {
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        throw std::runtime_error("Invalid matrix data type. Expected double.");
    }

    // Verify rowIndex is valid
    if (rowIndex >= matvar->dims[0])
    {
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        throw std::runtime_error("Row index out of bounds");
    }

    // Get data pointer
    auto* data = static_cast<double*>(matvar->data);
    if (data == nullptr)
    {
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        throw std::runtime_error("Failed to access data");
    }

    // Create Eigen vector for the audio samples
    Eigen::VectorXf audioSamples(AUDIO_SAMPLES_COUNT);

    // Copy audio samples
    for (int i = 0; i < AUDIO_SAMPLES_COUNT; ++i)
    {
        audioSamples(i) = static_cast<float>(data[i * matvar->dims[0] + rowIndex]);
    }

    // Get the EEG index
    int eegIndex = static_cast<int>(data[EEG_INDEX_COLUMN * matvar->dims[0] + rowIndex]);

    // Cleanup
    Mat_VarFree(matvar);
    Mat_Close(matfp);

    return {std::move(audioSamples), eegIndex};
}

} // namespace nn::dataLoaders

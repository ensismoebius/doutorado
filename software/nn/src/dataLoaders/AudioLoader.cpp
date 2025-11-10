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

    // Read first variable (assumed to be the audio data matrix)
    matvar_t* matvar = Mat_VarRead(matfp, NULL); // Read first variable
    if (matvar == nullptr)
    {
        Mat_Close(matfp);
        throw std::runtime_error("Failed to read variable from MAT file");
    }

    // Verify dimensions (M_rows x 176402)
    if (matvar->rank != 2 || matvar->dims[1] != 176402)
    {
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        throw std::runtime_error("Invalid matrix dimensions. Expected Mx176402");
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

    // Create Eigen vector for the audio samples (176400 samples)
    Eigen::VectorXf audioSamples(176400);

    // Copy audio samples (first 176400 columns)
    size_t offset = rowIndex * matvar->dims[1]; // Offset to the correct row
    for (int i = 0; i < 176400; ++i)
    {
        audioSamples(i) = static_cast<float>(data[offset + i]);
    }

    // Get the EEG index (last column, 176402)
    int eegIndex = static_cast<int>(data[offset + 176401]);

    // Cleanup
    Mat_VarFree(matvar);
    Mat_Close(matfp);

    return {std::move(audioSamples), eegIndex};
}

} // namespace nn::dataLoaders

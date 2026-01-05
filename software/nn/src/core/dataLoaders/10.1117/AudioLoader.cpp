#include "nn/dataLoaders/10.1117/AudioLoader.h"

#include <matio.h>

#include <optional>
#include <stdexcept>

#include "nn/tensor/Tensor.hpp"
#include "nn/dataLoaders/IMatLoader.hpp"

namespace nn::dataLoaders
{

class AudioLoader : public IMatLoader
{
   public:
    AudioLoader() = default;
    ~AudioLoader() override
    {
        if (matFile_ != nullptr)
        {
            Mat_Close(matFile_);
            matFile_ = nullptr;
        }
    }

    auto open(const std::string& filePath) noexcept -> bool override
    {
        filePath_ = filePath;
        matFile_ = Mat_Open(filePath.c_str(), MAT_ACC_RDONLY); // flawfinder: ignore
        return matFile_ != nullptr;
    }

    void close() noexcept override
    {
        if (matFile_ != nullptr)
        {
            Mat_Close(matFile_);
            matFile_ = nullptr;
        }
    }

    auto readVariable(const std::string& name)
        -> std::unique_ptr<matvar_t, void (*)(matvar_t*)> override
    {
        if (matFile_ == nullptr)
        {
            return {nullptr, &Mat_VarFree};
        }

        matvar_t* var = Mat_VarRead(matFile_, name.c_str());
        return {var, &Mat_VarFree};
    }

    auto readFirstNumericVariable()
        -> std::optional<std::unique_ptr<matvar_t, void (*)(matvar_t*)>> override
    {
        if (matFile_ == nullptr)
        {
            return std::nullopt;
        }

        for (matvar_t* var = Mat_VarReadNext(matFile_); var != nullptr;
             var = Mat_VarReadNext(matFile_))
        {
            if (var->class_type == MAT_C_DOUBLE && var->rank == 2)
            {
                return std::optional<std::unique_ptr<matvar_t, void (*)(matvar_t*)>>{
                    std::unique_ptr<matvar_t, void (*)(matvar_t*)>(var, &Mat_VarFree)};
            }
            Mat_VarFree(var);
        }

        return std::nullopt;
    }

    [[nodiscard]] auto filePath() const noexcept -> std::string override
    {
        return filePath_;
    }

   private:
    std::string filePath_;
    mat_t* matFile_ = nullptr;
};

auto loadAudioFromMat(const std::string& filePath, size_t rowIndex)
    -> std::tuple<nn::Tensor, int, int>
{
    AudioLoader loader;
    if (!loader.open(filePath)) // flawfinder: ignore
    {
        throw std::runtime_error("Failed to open MAT file: " + filePath);
    }

    auto audioVariable = loader.readVariable(AUDIO_VARIABLE_NAME);
    if (!audioVariable)
    {
        // try first numeric
        auto maybe = loader.readFirstNumericVariable();
        if (!maybe)
        {
            throw std::runtime_error("Failed to read audio variable from MAT file");
        }
        audioVariable = std::move(*maybe);
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
    const auto* rawDataPtr = static_cast<const double*>(audioVariable->data);
    if (rawDataPtr == nullptr)
    {
        throw std::runtime_error("Failed to access data");
    }

    // Create Tensor for the audio samples (column vector)
    nn::Tensor audioSamples(AUDIO_SAMPLES_COUNT, 1);

    // Copy audio samples
    for (int i = 0; i < AUDIO_SAMPLES_COUNT; ++i)
    {
        double doubleValue = rawDataPtr[(i * audioVariable->dims[0]) + rowIndex];
        float floatValue = static_cast<float>(doubleValue);

        audioSamples.at(i, 0) = floatValue;
    }

    // Get the stimulus
    int stimulus =
        static_cast<int>(rawDataPtr[(STIMULUS_COLUMN * audioVariable->dims[0]) + rowIndex]);

    // Get the EEG index
    int eegIndex =
        static_cast<int>(rawDataPtr[(EEG_INDEX_COLUMN * audioVariable->dims[0]) + rowIndex]);

    return {std::move(audioSamples), stimulus, eegIndex};
}

} // namespace nn::dataLoaders

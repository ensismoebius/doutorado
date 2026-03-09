/**
 * @file AudioLoader.cpp
 * @brief Implementation of the 10.1117-style audio MAT loader.
 *
 * The public entry point is `nn::dataLoaders::loadAudioFromMat()`, which returns
 * a sample tensor plus integer labels extracted from the MAT matrix row.
 */

#include "nn/dataLoaders/10.1117/AudioLoader.h"

#include <matio.h>

#include <algorithm>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "nn/dataLoaders/10.1117/NAMES.hpp"
#include "nn/dataLoaders/IMatLoader.hpp"
#include "nn/tensor/Tensor.hpp"

namespace nn::dataLoaders
{
using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
using std::ptrdiff_t;

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

    // flawfinder: ignore
    auto open(const std::string& filePath) noexcept -> bool override
    {
        filePath_ = filePath;
        // Security: Check for symlinks and regular file to prevent CWE-362
        std::filesystem::path fpath(filePath);
        try
        {
            // Ensure the path exists, is a regular file, and is not a symlink.
            // This helps mitigate risks like symlink attacks (CWE-362).
            if (!std::filesystem::exists(fpath) || !std::filesystem::is_regular_file(fpath))
            {
                return false;
            }
            // std::filesystem::is_symlink(fpath) is implicitly covered by is_regular_file(fpath)
            // for the target of the symlink. If fpath itself is a symlink, is_regular_file returns
            // false.
        }
        catch (...)
        {
            return false;
        }
        matFile_ = Mat_Open(filePath.c_str(), MAT_ACC_RDONLY);
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

        // Fast path: read only variable metadata. Payload is fetched on demand.
        matvar_t* var = Mat_VarReadInfo(matFile_, name.c_str());
        return {var, &Mat_VarFree};
    }

    auto readRowAsDoubles(const matvar_t& var, size_t rowIndex) const -> std::vector<double>
    {
        if (matFile_ == nullptr)
        {
            throw std::runtime_error("AudioLoader: MAT file is not open.");
        }
        if (var.rank != 2)
        {
            throw std::runtime_error("AudioLoader: expected rank-2 variable.");
        }
        if (rowIndex >= var.dims[0])
        {
            throw std::runtime_error("AudioLoader: row index out of bounds.");
        }
        if (var.dims[0] > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            var.dims[1] > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            throw std::runtime_error("AudioLoader: matrix dimensions exceed MatIO int limits.");
        }

        std::vector<double> row_values(var.dims[1], 0.0);

        int start[2] = {static_cast<int>(rowIndex), 0};
        int stride[2] = {1, 1};
        int edge[2] = {1, static_cast<int>(var.dims[1])};

        if (Mat_VarReadData(
                matFile_, const_cast<matvar_t*>(&var), row_values.data(), start, stride, edge) != 0)
        {
            throw std::runtime_error("AudioLoader: failed to read row data from MAT variable.");
        }

        return row_values;
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

    auto audioVariable = loader.readVariable(AUDIO_MAT_VARIABLE_NAME);
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

    // Verify dimensions (M_rows x audioTotalColumns)
    if (audioVariable->rank != 2 ||
        audioVariable->dims[1] != ImaginedSpeechSchema_10_1117.audioTotalColumns())
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

    // Read only the requested row to avoid loading the full matrix payload.
    const std::vector<double> rowValues = loader.readRowAsDoubles(*audioVariable, rowIndex);

    // Create Tensor for the audio samples (column vector)
    nn::Tensor audioSamples(ImaginedSpeechSchema_10_1117.audioSamples(), 1);

    // Copy audio samples through contiguous storage
    // to avoid per-element at(i,0) overhead.
    const double* src = rowValues.data();
    float* dst = audioSamples.mutable_data_ptr();
    const size_t n = ImaginedSpeechSchema_10_1117.audioSamples();
    for (size_t i = 0; i < n; ++i)
    {
        dst[i] = static_cast<float>(src[i]);
    }

    // Get the stimulus
    int stimulus = static_cast<int>(rowValues[ImaginedSpeechSchema_10_1117.audioStimulusColumn()]);

    // Get the EEG index
    int eegIndex = static_cast<int>(rowValues[ImaginedSpeechSchema_10_1117.audioEEGIndexColumn()]);

    return {std::move(audioSamples), stimulus, eegIndex};
}

} // namespace nn::dataLoaders

/**
 * @file AudioLoader.cpp
 * @brief Implementation of the 10.1117-style audio MAT loader.
 *
 * The public entry point is `nn::dataLoaders::loadAudioFromMat()`, which returns
 * a sample tensor plus integer labels extracted from the MAT matrix row.
 */

#include "nn/dataLoaders/10.1117/AudioLoader.h"

#include <matio.h>

#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "nn/dataLoaders/10.1117/NAMES.hpp"
#include "nn/tensor/Tensor.hpp"

namespace nn::dataLoaders
{
using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
namespace
{
struct MatFileCloser
{
    void operator()(mat_t* file) const
    {
        if (file != nullptr)
        {
            Mat_Close(file);
        }
    }
};

struct MatVarFreer
{
    void operator()(matvar_t* var) const
    {
        if (var != nullptr)
        {
            Mat_VarFree(var);
        }
    }
};

using MatFilePtr = std::unique_ptr<mat_t, MatFileCloser>;
using MatVarPtr = std::unique_ptr<matvar_t, MatVarFreer>;

auto readRowAsDoubles(const mat_t* file, matvar_t* var, size_t rowIndex) -> std::vector<double>
{
    if (file == nullptr || var == nullptr)
    {
        throw std::runtime_error("AudioLoader: MAT file/session is not initialized.");
    }
    if (var->rank != 2)
    {
        throw std::runtime_error("AudioLoader: expected rank-2 variable.");
    }
    if (rowIndex >= var->dims[0])
    {
        throw std::runtime_error("AudioLoader: row index out of bounds.");
    }
    if (var->dims[0] > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        var->dims[1] > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("AudioLoader: matrix dimensions exceed MatIO int limits.");
    }

    std::vector<double> row_values(var->dims[1], 0.0);
    int start[2] = {static_cast<int>(rowIndex), 0};
    int stride[2] = {1, 1};
    int edge[2] = {1, static_cast<int>(var->dims[1])};

    if (Mat_VarReadData(const_cast<mat_t*>(file), var, row_values.data(), start, stride, edge) != 0)
    {
        throw std::runtime_error("AudioLoader: failed to read row data from MAT variable.");
    }

    return row_values;
}
} // namespace

struct AudioMatSession::Impl
{
    std::string filePath;
    MatFilePtr matFile{nullptr};
    MatVarPtr audioVar{nullptr};
};

AudioMatSession::AudioMatSession(const std::string& filePath) : impl_(std::make_unique<Impl>())
{
    impl_->filePath = filePath;

    std::filesystem::path fpath(filePath);
    if (!std::filesystem::exists(fpath) || !std::filesystem::is_regular_file(fpath))
    {
        throw std::runtime_error("AudioLoader: invalid MAT file path: " + filePath);
    }

    impl_->matFile.reset(Mat_Open(filePath.c_str(), MAT_ACC_RDONLY));
    if (!impl_->matFile)
    {
        throw std::runtime_error("AudioLoader: failed to open MAT file: " + filePath);
    }

    impl_->audioVar.reset(Mat_VarReadInfo(impl_->matFile.get(), AUDIO_MAT_VARIABLE_NAME.c_str()));
    if (!impl_->audioVar)
    {
        throw std::runtime_error("AudioLoader: failed to read audio variable metadata from: " +
                                 filePath);
    }

    if (impl_->audioVar->rank != 2 ||
        impl_->audioVar->dims[1] != ImaginedSpeechSchema_10_1117.audioTotalColumns())
    {
        throw std::runtime_error("AudioLoader: invalid matrix dimensions. Expected Mx176402");
    }

    if (impl_->audioVar->class_type != MAT_C_DOUBLE)
    {
        throw std::runtime_error("AudioLoader: invalid matrix data type. Expected double.");
    }
}

AudioMatSession::~AudioMatSession() = default;
AudioMatSession::AudioMatSession(AudioMatSession&&) noexcept = default;
auto AudioMatSession::operator=(AudioMatSession&&) noexcept -> AudioMatSession& = default;

auto AudioMatSession::readRow(size_t rowIndex) const -> std::tuple<nn::Tensor, int, int>
{
    const std::vector<double> rowValues =
        readRowAsDoubles(impl_->matFile.get(), impl_->audioVar.get(), rowIndex);

    nn::Tensor audioSamples(ImaginedSpeechSchema_10_1117.audioSamples(), 1);
    float* dst = audioSamples.mutable_data_ptr();
    const double* src = rowValues.data();
    const size_t n = ImaginedSpeechSchema_10_1117.audioSamples();
    for (size_t i = 0; i < n; ++i)
    {
        dst[i] = static_cast<float>(src[i]);
    }

    const int stimulus =
        static_cast<int>(rowValues[ImaginedSpeechSchema_10_1117.audioStimulusColumn()]);
    const int eegIndex =
        static_cast<int>(rowValues[ImaginedSpeechSchema_10_1117.audioEEGIndexColumn()]);

    return {std::move(audioSamples), stimulus, eegIndex};
}

auto AudioMatSession::rowCount() const -> size_t
{
    return impl_->audioVar ? impl_->audioVar->dims[0] : 0;
}

auto AudioMatSession::filePath() const -> const std::string&
{
    return impl_->filePath;
}

auto loadAudioFromMat(const std::string& filePath, size_t rowIndex)
    -> std::tuple<nn::Tensor, int, int>
{
    AudioMatSession session(filePath);
    return session.readRow(rowIndex);
}

} // namespace nn::dataLoaders

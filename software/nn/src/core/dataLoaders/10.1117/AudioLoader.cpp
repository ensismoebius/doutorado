/**
 * @file AudioLoader.cpp
 * @brief Implementation of the 10.1117-style audio MAT loader.
 *
 * The public entry point is `nn::dataLoaders::loadAudioFromMat()`, which returns
 * a sample tensor plus integer labels extracted from the MAT matrix row.
 */

#include "nn/dataLoaders/10.1117/AudioLoader.h"

#include <matio.h>

#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
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

auto readRowsAsDoubles(const mat_t* file, matvar_t* var, size_t startRow, size_t rowCount)
    -> std::vector<double>
{
    if (file == nullptr || var == nullptr)
    {
        throw std::runtime_error("AudioLoader: MAT file/session is not initialized.");
    }
    if (var->rank != 2)
    {
        throw std::runtime_error("AudioLoader: expected rank-2 variable.");
    }
    if (rowCount == 0)
    {
        return {};
    }
    if (startRow >= var->dims[0] || (startRow + rowCount) > var->dims[0])
    {
        throw std::runtime_error("AudioLoader: row range out of bounds.");
    }
    if (var->dims[0] > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        var->dims[1] > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        rowCount > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("AudioLoader: matrix dimensions exceed MatIO int limits.");
    }

    std::vector<double> block_values(rowCount * var->dims[1], 0.0);
    int start[2] = {static_cast<int>(startRow), 0};
    int stride[2] = {1, 1};
    int edge[2] = {static_cast<int>(rowCount), static_cast<int>(var->dims[1])};

    if (Mat_VarReadData(const_cast<mat_t*>(file), var, block_values.data(), start, stride, edge) !=
        0)
    {
        throw std::runtime_error("AudioLoader: failed to read row block from MAT variable.");
    }

    return block_values;
}
} // namespace

struct AudioMatSession::Impl
{
    std::string filePath;
    MatFilePtr matFile{nullptr};
    MatVarPtr audioVar{nullptr};
    static constexpr size_t kRowCacheCapacity = 8;
    mutable std::unordered_map<size_t, std::tuple<nn::Tensor, int, int>> rowCache;
    mutable std::deque<size_t> rowCacheOrder;
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
    if (const auto it = impl_->rowCache.find(rowIndex); it != impl_->rowCache.end())
    {
        return it->second;
    }

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

    std::tuple<nn::Tensor, int, int> result{std::move(audioSamples), stimulus, eegIndex};

    if (impl_->rowCache.size() >= Impl::kRowCacheCapacity)
    {
        const size_t evict = impl_->rowCacheOrder.front();
        impl_->rowCacheOrder.pop_front();
        impl_->rowCache.erase(evict);
    }
    impl_->rowCache[rowIndex] = result;
    impl_->rowCacheOrder.push_back(rowIndex);

    return result;
}

auto AudioMatSession::readRows(size_t startRow, size_t rowCount) const
    -> std::vector<std::tuple<nn::Tensor, int, int>>
{
    if (rowCount == 0)
    {
        return {};
    }

    std::vector<std::tuple<nn::Tensor, int, int>> out;
    out.reserve(rowCount);

    bool allCached = true;
    for (size_t r = 0; r < rowCount; ++r)
    {
        const size_t row = startRow + r;
        const auto it = impl_->rowCache.find(row);
        if (it == impl_->rowCache.end())
        {
            allCached = false;
            break;
        }
    }

    if (allCached)
    {
        for (size_t r = 0; r < rowCount; ++r)
        {
            out.push_back(impl_->rowCache.at(startRow + r));
        }
        return out;
    }

    const std::vector<double> blockValues =
        readRowsAsDoubles(impl_->matFile.get(), impl_->audioVar.get(), startRow, rowCount);
    for (size_t r = 0; r < rowCount; ++r)
    {
        nn::Tensor audioSamples(ImaginedSpeechSchema_10_1117.audioSamples(), 1);
        float* dst = audioSamples.mutable_data_ptr();
        const size_t n = ImaginedSpeechSchema_10_1117.audioSamples();
        for (size_t i = 0; i < n; ++i)
        {
            dst[i] = static_cast<float>(blockValues[(i * rowCount) + r]);
        }

        const int stimulus = static_cast<int>(
            blockValues[(ImaginedSpeechSchema_10_1117.audioStimulusColumn() * rowCount) + r]);
        const int eegIndex = static_cast<int>(
            blockValues[(ImaginedSpeechSchema_10_1117.audioEEGIndexColumn() * rowCount) + r]);

        std::tuple<nn::Tensor, int, int> sample{std::move(audioSamples), stimulus, eegIndex};

        const size_t rowIndex = startRow + r;
        if (impl_->rowCache.size() >= Impl::kRowCacheCapacity)
        {
            const size_t evict = impl_->rowCacheOrder.front();
            impl_->rowCacheOrder.pop_front();
            impl_->rowCache.erase(evict);
        }
        impl_->rowCache[rowIndex] = sample;
        impl_->rowCacheOrder.push_back(rowIndex);

        out.emplace_back(std::move(sample));
    }

    return out;
}

auto AudioMatSession::readRowsFlat(size_t startRow, size_t rowCount) const -> AudioRowsFlat
{
    AudioRowsFlat out{};
    if (rowCount == 0)
    {
        return out;
    }

    const size_t samplesPerRow = ImaginedSpeechSchema_10_1117.audioSamples();
    const std::vector<double> blockValues =
        readRowsAsDoubles(impl_->matFile.get(), impl_->audioVar.get(), startRow, rowCount);

    out.samples.resize(rowCount * samplesPerRow);
    out.stimuli.resize(rowCount);
    out.eegIndices.resize(rowCount);

    for (size_t r = 0; r < rowCount; ++r)
    {
        for (size_t i = 0; i < samplesPerRow; ++i)
        {
            out.samples[(r * samplesPerRow) + i] =
                static_cast<float>(blockValues[(i * rowCount) + r]);
        }

        out.stimuli[r] = static_cast<int>(
            blockValues[(ImaginedSpeechSchema_10_1117.audioStimulusColumn() * rowCount) + r]);
        out.eegIndices[r] = static_cast<int>(
            blockValues[(ImaginedSpeechSchema_10_1117.audioEEGIndexColumn() * rowCount) + r]);
    }

    return out;
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

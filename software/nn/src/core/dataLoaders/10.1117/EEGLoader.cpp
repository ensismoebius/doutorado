/**
 * @file EEGLoader.cpp
 * @brief Implementation of the 10.1117-style EEG MAT loader.
 */

#include "nn/dataLoaders/10.1117/EEGLoader.h"

#include <matio.h>

#include <array>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "nn/dataLoaders/10.1117/NAMES.hpp"
#include "nn/tensor/Tensor.hpp"

/*
 * EEGLoader implementation notes
 * ----------------------------
 * This loader expects a MATLAB v5 double matrix with shape (N_rows x 24579). The
 * first 24576 columns are raw EEG samples and are interpreted as 6 channels × 4096
 * samples (contiguous blocks per channel). The last 3 columns are labels:
 *   - column index 24576 -> modality
 *   - column index 24577 -> stimulus
 *   - column index 24578 -> artifact
 *
 * The code reads the matrix as MatIO stores it (column-major with columns = features).
 * If your data stores samples interleaved across channels (time-major interleaving),
 * change the mapping logic near the "split into channels" comment: instead of
 * taking contiguous blocks per channel, distribute samples alternately into each
 * channel.
 */

namespace nn::dataLoaders
{
using std::string;

// Alias for unique_ptr to matvar_t with custom deleter
using MatVarUniquePtr = std::unique_ptr<matvar_t, void (*)(matvar_t*)>;

namespace
{
struct MatFileCloser
{
    void operator()(mat_t* f) const
    {
        if (f != nullptr)
        {
            Mat_Close(f);
        }
    }
};

struct MatVarFreer
{
    void operator()(matvar_t* v) const
    {
        if (v != nullptr)
        {
            Mat_VarFree(v);
        }
    }
};

using SessionMatFilePtr = std::unique_ptr<mat_t, MatFileCloser>;
using SessionMatVarPtr = std::unique_ptr<matvar_t, MatVarFreer>;

auto readMatRow(const mat_t* matFile, matvar_t* var, size_t rowIndex) -> std::vector<double>
{
    if (matFile == nullptr || var == nullptr)
    {
        throw std::runtime_error("EEGLoader: MAT file/session not initialized.");
    }
    if (var->rank != 2)
    {
        throw std::runtime_error("EEGLoader: expected rank-2 variable.");
    }
    if (rowIndex >= var->dims[0])
    {
        throw std::runtime_error("EEGLoader: row index out of bounds");
    }
    if (var->dims[0] > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        var->dims[1] > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("EEGLoader: matrix dimensions exceed MatIO int limits.");
    }

    std::vector<double> row_values(var->dims[1], 0.0);
    int start[2] = {static_cast<int>(rowIndex), 0};
    int stride[2] = {1, 1};
    int edge[2] = {1, static_cast<int>(var->dims[1])};

    if (Mat_VarReadData(const_cast<mat_t*>(matFile), var, row_values.data(), start, stride, edge) !=
        0)
    {
        throw std::runtime_error("Failed to read EEG row data from MAT variable");
    }

    return row_values;
}
} // namespace

struct EEGMatSession::Impl
{
    std::string filePath;
    SessionMatFilePtr matFile{nullptr};
    SessionMatVarPtr eegVar{nullptr};
};

EEGMatSession::EEGMatSession(const std::string& filePath) : impl_(std::make_unique<Impl>())
{
    impl_->filePath = filePath;

    std::filesystem::path fpath(filePath);
    if (!std::filesystem::exists(fpath) || !std::filesystem::is_regular_file(fpath) ||
        std::filesystem::is_symlink(fpath))
    {
        throw std::runtime_error("EEGLoader: invalid MAT file path: " + filePath);
    }

    impl_->matFile.reset(Mat_Open(filePath.c_str(), MAT_ACC_RDONLY));
    if (!impl_->matFile)
    {
        throw std::runtime_error("Failed to open MAT file: " + filePath);
    }

    impl_->eegVar.reset(Mat_VarReadInfo(impl_->matFile.get(), EEG_MAT_VARIABLE_NAME.c_str()));
    if (!impl_->eegVar)
    {
        throw std::runtime_error("Failed to find EEG variable in MAT file: " + filePath);
    }

    if (impl_->eegVar->rank != 2 ||
        impl_->eegVar->dims[1] != nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegTotalColumns())
    {
        throw std::runtime_error("Invalid EEG matrix dimensions. Expected Nx24579");
    }

    if (impl_->eegVar->class_type != MAT_C_DOUBLE)
    {
        throw std::runtime_error("Invalid EEG matrix data type. Expected double.");
    }
}

EEGMatSession::~EEGMatSession() = default;
EEGMatSession::EEGMatSession(EEGMatSession&&) noexcept = default;
auto EEGMatSession::operator=(EEGMatSession&&) noexcept -> EEGMatSession& = default;

auto EEGMatSession::readRow(size_t rowIndex) const -> std::tuple<nn::Tensor, std::array<int, 3>>
{
    std::vector<double> row_values =
        readMatRow(impl_->matFile.get(), impl_->eegVar.get(), rowIndex);

    nn::Tensor eegChannels(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels,
                           nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel());

    const size_t samplesPerChannel =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel();
    for (size_t ch = 0; ch < nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels; ++ch)
    {
        for (size_t s = 0; s < samplesPerChannel; ++s)
        {
            eegChannels.at(ch, s) = static_cast<float>(row_values[(ch * samplesPerChannel) + s]);
        }
    }

    const size_t modality_column = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegModeColumn();
    const size_t stimulus_column =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegStimulusColumn();
    const size_t artifact_column = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegBlinkColumn();

    int modality = static_cast<int>(row_values[modality_column]);
    int stimulus = static_cast<int>(row_values[stimulus_column]);
    int artifact = static_cast<int>(row_values[artifact_column]);

    return {std::move(eegChannels), {modality, stimulus, artifact}};
}

auto EEGMatSession::rowCount() const -> size_t
{
    return impl_->eegVar ? impl_->eegVar->dims[0] : 0;
}

auto EEGMatSession::filePath() const -> const std::string&
{
    return impl_->filePath;
}

auto EEGLoader::readFirstNumericVariable() -> std::optional<MatVarUniquePtr>
{
    if (matFile_ == nullptr)
    {
        return std::nullopt;
    }

    // Iterate variables looking for numeric matrix
    for (matvar_t* var = Mat_VarReadNextInfo(matFile_); var != nullptr;
         var = Mat_VarReadNextInfo(matFile_))
    {
        if (var->class_type == MAT_C_DOUBLE && var->rank == 2)
        {
            // Rewind is not trivial; return the var ownership to caller
            return MatVarUniquePtr(var, &Mat_VarFree);
        }
        Mat_VarFree(var);
    }

    return std::nullopt;
}

// flawfinder: ignore
auto EEGLoader::open(const std::string& filePath) noexcept -> bool
{
    filePath_ = filePath;
    // Security: Check for symlinks and regular file
    std::filesystem::path fpath(filePath);
    try
    {
        if (!std::filesystem::exists(fpath) || !std::filesystem::is_regular_file(fpath) ||
            std::filesystem::is_symlink(fpath))
        {
            return false;
        }
    }
    catch (...)
    {
        return false;
    }
    matFile_ = Mat_Open(filePath.c_str(), MAT_ACC_RDONLY); // flawfinder: ignore
    return matFile_ != nullptr;
}

void EEGLoader::close() noexcept
{
    if (matFile_ != nullptr)
    {
        Mat_Close(matFile_);
        matFile_ = nullptr;
    }
}

auto EEGLoader::readVariable(const std::string& name)
    -> std::unique_ptr<matvar_t, void (*)(matvar_t*)>
{
    if (matFile_ == nullptr)
    {
        return {nullptr, &Mat_VarFree};
    }

    // Fast path: load variable metadata only; data is read on demand.
    matvar_t* var = Mat_VarReadInfo(matFile_, name.c_str());
    return {var, &Mat_VarFree};
}

auto loadEEGFromMat(const std::string& filePath, size_t rowIndex)
    -> std::tuple<nn::Tensor, std::array<int, 3>>
{
    EEGMatSession session(filePath);
    return session.readRow(rowIndex);
}
} // namespace nn::dataLoaders

// Provide out-of-line destructor definition so the vtable/typeinfo is emitted
// (declared `~EEGLoader()` in the header with `override`).
nn::dataLoaders::EEGLoader::~EEGLoader() = default;

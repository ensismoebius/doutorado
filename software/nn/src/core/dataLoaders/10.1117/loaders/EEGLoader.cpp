/**
 * @file EEGLoader.cpp
 * @brief Implementation of the 10.1117-style EEG MAT loader.
 */

#include "nn/dataLoaders/10.1117/loaders/EEGLoader.h"

#include <matio.h>
#include <sqlite3.h>

#include <array>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "nn/dataLoaders/10.1117/schema/METADATA.hpp"
#include "nn/dataLoaders/10.1117/schema/NAMES.hpp"
#include "nn/dataLoaders/10.1117/schema/SchemaIndexing.hpp"
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
using nn::dataLoaders::schema101117::columnMajorIndex;
using nn::dataLoaders::schema101117::eegSignalFlatColumn;
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

auto readMatRows(const mat_t* matFile, matvar_t* var, size_t startRow, size_t rowCount)
    -> std::vector<double>
{
    if (matFile == nullptr || var == nullptr)
    {
        throw std::runtime_error("EEGLoader: MAT file/session not initialized.");
    }
    if (var->rank != 2)
    {
        throw std::runtime_error("EEGLoader: expected rank-2 variable.");
    }
    if (rowCount == 0)
    {
        return {};
    }
    if (startRow >= var->dims[0] || (startRow + rowCount) > var->dims[0])
    {
        throw std::runtime_error("EEGLoader: row range out of bounds");
    }
    if (var->dims[0] > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        var->dims[1] > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        rowCount > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("EEGLoader: matrix dimensions exceed MatIO int limits.");
    }

    std::vector<double> block_values(rowCount * var->dims[1], 0.0);
    int start[2] = {static_cast<int>(startRow), 0};
    int stride[2] = {1, 1};
    int edge[2] = {static_cast<int>(rowCount), static_cast<int>(var->dims[1])};

    if (Mat_VarReadData(
            const_cast<mat_t*>(matFile), var, block_values.data(), start, stride, edge) != 0)
    {
        throw std::runtime_error("Failed to read EEG row block from MAT variable");
    }

    return block_values;
}
} // namespace

struct EEGMatSession::Impl
{
    std::string filePath;
    SessionMatFilePtr matFile{nullptr};
    SessionMatVarPtr eegVar{nullptr};
    static constexpr size_t kRowCacheCapacity = 32;
    mutable std::unordered_map<size_t, std::tuple<nn::Tensor, std::array<int, 3>>> rowCache;
    mutable std::deque<size_t> rowCacheOrder;
    bool is_shard = false;
    // sqlite support
    bool is_sqlite = false;
    sqlite3* db = nullptr;
    int subject_id = -1;
};

EEGMatSession::EEGMatSession(const std::string& filePath, int subject_id)
    : impl_(std::make_unique<Impl>())
{
    impl_->filePath = filePath;
    impl_->subject_id = subject_id;

    // Detect sqlite DB path by extension and try opening read-only.
    try
    {
        if (!filePath.empty() && filePath.size() > 7 &&
            filePath.substr(filePath.size() - 7) == ".sqlite")
        {
            if (sqlite3_open_v2(filePath.c_str(), &impl_->db, SQLITE_OPEN_READONLY, nullptr) ==
                SQLITE_OK)
            {
                impl_->is_sqlite = true;
            }
            else if (impl_->db)
            {
                sqlite3_close(impl_->db);
                impl_->db = nullptr;
            }
        }
    }
    catch (...)
    { /* fall back to MAT below */
    }

    if (!impl_->is_sqlite)
    {
        std::filesystem::path fpath(filePath);
        if (!std::filesystem::exists(fpath) || !std::filesystem::is_regular_file(fpath))
        {
            throw std::runtime_error("EEGLoader: invalid file path: " + filePath);
        }

        impl_->matFile.reset(Mat_Open(filePath.c_str(), MAT_ACC_RDONLY));
        if (!impl_->matFile)
        {
            throw std::runtime_error("EEGLoader: failed to open MAT file: " + filePath);
        }

        impl_->eegVar.reset(Mat_VarReadInfo(impl_->matFile.get(), kEegMatVariableName.c_str()));
        if (!impl_->eegVar)
        {
            throw std::runtime_error(
                "EEGLoader: failed to read EEG variable metadata from: " + filePath);
        }

        if (impl_->eegVar->rank != 2 ||
            impl_->eegVar->dims[1] !=
                nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegTotalColumns())
        {
            throw std::runtime_error("EEGLoader: invalid matrix dimensions. Expected Nx24579");
        }

        if (impl_->eegVar->class_type != MAT_C_DOUBLE)
        {
            throw std::runtime_error("EEGLoader: invalid matrix data type. Expected double.");
        }
    }
}

EEGMatSession::~EEGMatSession()
{
    if (impl_ && impl_->is_sqlite && impl_->db)
    {
        sqlite3_close(impl_->db);
        impl_->db = nullptr;
    }
}
EEGMatSession::EEGMatSession(EEGMatSession&&) noexcept = default;
auto EEGMatSession::operator=(EEGMatSession&&) noexcept -> EEGMatSession& = default;

auto EEGMatSession::readRow(size_t rowIndex) const -> std::tuple<nn::Tensor, std::array<int, 3>>
{
    if (const auto it = impl_->rowCache.find(rowIndex); it != impl_->rowCache.end())
    {
        return it->second;
    }

    if (impl_->is_sqlite)
    {
        if (!impl_->db || impl_->subject_id < 0)
        {
            throw std::runtime_error("EEGLoader(SQL): DB not initialized with subject id");
        }

        const char* sql =
            "SELECT e.F3, e.F4, e.C3, e.C4, e.P3, e.P4, e.blink, t.modality_id, t.stimulus_id FROM "
            "eeg_samples e JOIN trial t ON e.trial_id = t.id WHERE t.subject_id = ? AND "
            "t.original_row = ? LIMIT 1";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        {
            throw std::runtime_error("EEGLoader(SQL): failed to prepare statement");
        }
        if (sqlite3_bind_int(stmt, 1, impl_->subject_id) != SQLITE_OK ||
            sqlite3_bind_int64(stmt, 2, static_cast<sqlite3_int64>(rowIndex)) != SQLITE_OK)
        {
            sqlite3_finalize(stmt);
            throw std::runtime_error("EEGLoader(SQL): failed to bind parameters");
        }

        nn::Tensor eegChannels(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels,
            nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel());

        int modality = 0;
        int stimulus = 0;
        int artifact = 0;

        if (sqlite3_step(stmt) == SQLITE_ROW)
        {
            // For each channel column (0..5) read blob of doubles
            for (int ch = 0; ch < 6; ++ch)
            {
                const void* blob = sqlite3_column_blob(stmt, ch);
                int bytes = sqlite3_column_bytes(stmt, ch);
                const size_t expected_bytes =
                    nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel() *
                    sizeof(double);
                if (static_cast<size_t>(bytes) != expected_bytes)
                {
                    sqlite3_finalize(stmt);
                    throw std::runtime_error("EEGLoader(SQL): unexpected eeg channel blob size");
                }
                const double* src = reinterpret_cast<const double*>(blob);
                for (size_t s = 0;
                    s < static_cast<size_t>(
                            nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel());
                    ++s)
                {
                    eegChannels.at(ch, s) = static_cast<float>(src[s]);
                }
            }

            artifact = sqlite3_column_int(stmt, 6);
            if (sqlite3_column_type(stmt, 7) != SQLITE_NULL) modality = sqlite3_column_int(stmt, 7);
            if (sqlite3_column_type(stmt, 8) != SQLITE_NULL) stimulus = sqlite3_column_int(stmt, 8);
        }
        else
        {
            sqlite3_finalize(stmt);
            throw std::runtime_error("EEGLoader(SQL): eeg row not found for subject");
        }

        sqlite3_finalize(stmt);

        std::tuple<nn::Tensor, std::array<int, 3>> result{
            std::move(eegChannels), std::array<int, 3>{modality, stimulus, artifact}};

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

    // MAT-backed path
    std::vector<double> row_values;
    row_values = readMatRow(impl_->matFile.get(), impl_->eegVar.get(), rowIndex);

    nn::Tensor eegChannels(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels,
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel());

    const size_t samplesPerChannel =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel();
    for (size_t ch = 0; ch < nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels; ++ch)
    {
        for (size_t s = 0; s < samplesPerChannel; ++s)
        {
            eegChannels.at(ch, s) = static_cast<float>(row_values[eegSignalFlatColumn(ch, s)]);
        }
    }

    const size_t modality_column = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegModeColumn();
    const size_t stimulus_column =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegStimulusColumn();
    const size_t artifact_column = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegBlinkColumn();

    int modality = static_cast<int>(row_values[modality_column]);
    int stimulus = static_cast<int>(row_values[stimulus_column]);
    int artifact = static_cast<int>(row_values[artifact_column]);

    std::tuple<nn::Tensor, std::array<int, 3>> result{
        std::move(eegChannels), std::array<int, 3>{modality, stimulus, artifact}};

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

auto EEGMatSession::readRows(size_t startRow, size_t rowCount) const
    -> std::vector<std::tuple<nn::Tensor, std::array<int, 3>>>
{
    if (rowCount == 0)
    {
        return {};
    }

    std::vector<std::tuple<nn::Tensor, std::array<int, 3>>> out;
    out.reserve(rowCount);

    if (impl_->is_sqlite)
    {
        std::vector<std::tuple<nn::Tensor, std::array<int, 3>>> _out;
        _out.reserve(rowCount);
        for (size_t r = 0; r < rowCount; ++r)
        {
            _out.emplace_back(readRow(startRow + r));
        }
        return _out;
    }

    const size_t samplesPerChannel =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel();
    const size_t totalCols = impl_->eegVar->dims[1];

    const std::vector<double> blockValues =
        readMatRows(impl_->matFile.get(), impl_->eegVar.get(), startRow, rowCount);

    for (size_t r = 0; r < rowCount; ++r)
    {
        nn::Tensor eegChannels(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels,
            nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel());

        for (size_t ch = 0; ch < nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels; ++ch)
        {
            for (size_t s = 0; s < samplesPerChannel; ++s)
            {
                eegChannels.at(ch, s) = static_cast<float>(
                    blockValues[eegSignalFlatColumn(ch, s + r * samplesPerChannel)]);
            }
        }

        const int modality = static_cast<int>(blockValues[columnMajorIndex(
            nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegModeColumn(), r, rowCount)]);
        const int stimulus = static_cast<int>(blockValues[columnMajorIndex(
            nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegStimulusColumn(), r, rowCount)]);
        const int artifact = static_cast<int>(blockValues[columnMajorIndex(
            nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegBlinkColumn(), r, rowCount)]);

        std::tuple<nn::Tensor, std::array<int, 3>> sample{
            std::move(eegChannels), std::array<int, 3>{modality, stimulus, artifact}};

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

auto EEGMatSession::readRowsFlat(size_t startRow, size_t rowCount) const -> EEGRowsFlat
{
    EEGRowsFlat out{};
    if (rowCount == 0)
    {
        return out;
    }
    if (impl_->is_sqlite)
    {
        const size_t signalCols = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns();
        out.signals.resize(rowCount * signalCols);
        out.labels.resize(rowCount);
        for (size_t r = 0; r < rowCount; ++r)
        {
            const auto tup = readRow(startRow + r);
            const auto& tensor = std::get<0>(tup);
            const auto labels = std::get<1>(tup);
            // copy flat signals
            const size_t ch = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels;
            const size_t samples =
                nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel();
            for (size_t c = 0; c < ch; ++c)
            {
                for (size_t s = 0; s < samples; ++s)
                {
                    out.signals[(r * signalCols) + (c * samples) + s] = tensor.at(c, s);
                }
            }
            out.labels[r] = labels;
        }
        return out;
    }

    // MAT-backed flat read
    const std::vector<double> block_values =
        readMatRows(impl_->matFile.get(), impl_->eegVar.get(), startRow, rowCount);

    const size_t signalCols = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns();
    out.signals.resize(rowCount * signalCols);
    out.labels.resize(rowCount);

    const size_t modality_column = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegModeColumn();
    const size_t stimulus_column =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegStimulusColumn();
    const size_t artifact_column = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegBlinkColumn();

    for (size_t r = 0; r < rowCount; ++r)
    {
        for (size_t c = 0; c < signalCols; ++c)
        {
            out.signals[(r * signalCols) + c] =
                static_cast<float>(block_values[columnMajorIndex(c, r, rowCount)]);
        }

        out.labels[r] = std::array<int, 3>{
            static_cast<int>(block_values[columnMajorIndex(modality_column, r, rowCount)]),
            static_cast<int>(block_values[columnMajorIndex(stimulus_column, r, rowCount)]),
            static_cast<int>(block_values[columnMajorIndex(artifact_column, r, rowCount)])};
    }

    return out;
}

auto EEGMatSession::rowCount() const -> size_t
{
    if (impl_->is_sqlite)
    {
        if (!impl_->db || impl_->subject_id < 0) return 0;
        const char* sql =
            "SELECT COUNT(*) FROM trial WHERE subject_id = ? AND original_row IS NOT NULL";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) return 0;
        sqlite3_bind_int(stmt, 1, impl_->subject_id);
        size_t count = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW)
        {
            count = static_cast<size_t>(sqlite3_column_int64(stmt, 0));
        }
        sqlite3_finalize(stmt);
        return count;
    }

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

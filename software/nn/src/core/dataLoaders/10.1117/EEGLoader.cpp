#include "EEGLoader.h"

#include <matio.h>

#include <array>
#include <memory>
#include <stdexcept>

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

// Constants matching dataset schema
static constexpr int EEG_TOTAL_COLUMNS = 24579; // M columns including labels
static constexpr int EEG_SAMPLE_COUNT = 24576;  // samples portion
static constexpr int EEG_CHANNELS = 6;          // number of EEG channels

// Alias for unique_ptr to matvar_t with custom deleter
using MatVarUniquePtr = std::unique_ptr<matvar_t, void (*)(matvar_t*)>;

auto EEGLoader::readFirstNumericVariable() -> std::optional<MatVarUniquePtr>
{
    if (matFile_ == nullptr)
    {
        return std::nullopt;
    }

    // Iterate variables looking for numeric matrix
    for (matvar_t* var = Mat_VarReadNext(matFile_); var != nullptr; var = Mat_VarReadNext(matFile_))
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

auto EEGLoader::open(const std::string& filePath) noexcept -> bool
{
    filePath_ = filePath;
    matFile_ = Mat_Open(filePath.c_str(), MAT_ACC_RDONLY);
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

    matvar_t* var = Mat_VarRead(matFile_, name.c_str());
    return {var, &Mat_VarFree};
}

auto loadEEGFromMat(const std::string& filePath, size_t rowIndex)
    -> std::tuple<Eigen::MatrixXf, std::array<int, 3>>
{
    // Open MAT file with RAII
    auto matFileDeleter = [](mat_t* mat) { Mat_Close(mat); };
    std::unique_ptr<mat_t, decltype(matFileDeleter)> matFile(
        Mat_Open(filePath.c_str(), MAT_ACC_RDONLY), matFileDeleter);

    if (!matFile)
    {
        throw std::runtime_error("Failed to open MAT file: " + filePath);
    }

    // Try to find a variable named "EEG" first
    matvar_t* var = Mat_VarRead(matFile.get(), "EEG");

    MatVarUniquePtr eegVar(var, &Mat_VarFree);

    // If not found, try to find the first numeric variable
    if (!eegVar)
    {
        // iterate variables
        for (matvar_t* v = Mat_VarReadNext(matFile.get()); v != nullptr;
             v = Mat_VarReadNext(matFile.get()))
        {
            if (v->class_type == MAT_C_DOUBLE && v->rank == 2)
            {
                eegVar.reset(v);
                break;
            }
            Mat_VarFree(v);
        }
    }

    if (!eegVar)
    {
        throw std::runtime_error("Failed to find EEG variable in MAT file: " + filePath);
    }

    // Validate dims
    if (eegVar->rank != 2 || eegVar->dims[1] != EEG_TOTAL_COLUMNS)
    {
        throw std::runtime_error("Invalid EEG matrix dimensions. Expected Nx24579");
    }

    if (eegVar->class_type != MAT_C_DOUBLE)
    {
        throw std::runtime_error("Invalid EEG matrix data type. Expected double.");
    }

    if (rowIndex >= eegVar->dims[0])
    {
        throw std::runtime_error("Row index out of bounds");
    }

    const auto* rawDataPtr = static_cast<const double*>(eegVar->data);
    if (rawDataPtr == nullptr)
    {
        throw std::runtime_error("Failed to access EEG data pointer");
    }

    // We'll construct an Eigen matrix with rows = channels, cols = samples_per_channel
    Eigen::MatrixXf eegChannels(EEG_CHANNELS, EEG_SAMPLE_COUNT / EEG_CHANNELS);

    // The dataset in documentations says: 6 channels × 4096 samples = 24576 samples
    // We'll assume the samples are interleaved per channel in blocks (channel-major or
    // sample-major?) The AudioLoader uses column-major indexing: rawData[(i * rows) + rowIndex]
    // That implies the MAT file stores data with rows = observations, columns = features. So each
    // column corresponds to a feature across rows. Here, columns are 24579 features: first 24576
    // are samples. We'll extract them and reshape into channels.

    const int rows = static_cast<int>(eegVar->dims[0]);

    // Extract sample values into a temporary vector of length EEG_SAMPLE_COUNT
    std::vector<float> samples;
    samples.reserve(EEG_SAMPLE_COUNT);
    for (int i = 0; i < EEG_SAMPLE_COUNT; ++i)
    {
        double v = rawDataPtr[(i * rows) + static_cast<int>(rowIndex)];
        samples.push_back(static_cast<float>(v));
    }

    // Now, split into channels. We'll assume contiguous blocks per channel: channel 0 samples at
    // indices 0..4095, channel1 4096..8191, etc.
    const int samplesPerChannel = EEG_SAMPLE_COUNT / EEG_CHANNELS;
    for (int ch = 0; ch < EEG_CHANNELS; ++ch)
    {
        for (int s = 0; s < samplesPerChannel; ++s)
        {
            eegChannels(ch, s) = samples[(ch * samplesPerChannel) + s];
        }
    }

    // Read labels from the last three columns
    int baseIdx = EEG_SAMPLE_COUNT; // index of first label column
    int modality = static_cast<int>(rawDataPtr[(baseIdx * rows) + static_cast<int>(rowIndex)]);
    int stimulus =
        static_cast<int>(rawDataPtr[((baseIdx + 1) * rows) + static_cast<int>(rowIndex)]);
    int artifact =
        static_cast<int>(rawDataPtr[((baseIdx + 2) * rows) + static_cast<int>(rowIndex)]);

    return {eegChannels, {modality, stimulus, artifact}};
}

} // namespace nn::dataLoaders

// Provide out-of-line destructor definition so the vtable/typeinfo is emitted
// (declared `~EEGLoader()` in the header with `override`).
nn::dataLoaders::EEGLoader::~EEGLoader() = default;

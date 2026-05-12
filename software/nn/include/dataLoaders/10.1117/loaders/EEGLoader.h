/**
 * @file EEGLoader.h
 * @brief MAT-file loader for EEG rows in the 10.1117-style dataset.
 *
 * The public API here is split into:
 * - `EEGLoader` (stateful) to open/close and read MAT variables.
 * - `loadEEGFromMat()` (stateless convenience) returning `nn::Tensor` + label array.
 */

// EEGLoader.h
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "dataLoaders/interfaces/IMatLoader.hpp"
#include "tensor/Tensor.hpp"

namespace nn::dataLoaders
{

struct EEGRowsFlat
{
    std::vector<float> signals; // row-major: [row0_signal..., row1_signal...]
    std::vector<std::array<int, 3>> labels;
};

class EEGMatSession
{
   public:
    // `filePath` may be either a path to a .mat file or a .sqlite database.
    // When using a sqlite DB, callers should provide `subject_id` so the
    // session can scope queries to a single subject. A value of -1 means
    // "no subject scope" (used for MAT-file mode).
    explicit EEGMatSession(const std::string& filePath, int subject_id = -1);
    ~EEGMatSession();

    EEGMatSession(const EEGMatSession&) = delete;
    EEGMatSession& operator=(const EEGMatSession&) = delete;
    EEGMatSession(EEGMatSession&&) noexcept;
    EEGMatSession& operator=(EEGMatSession&&) noexcept;

    auto readRow(size_t rowIndex) const -> std::tuple<nn::Tensor, std::array<int, 3>>;
    auto readRowsFlat(size_t startRow, size_t rowCount) const -> EEGRowsFlat;
    auto readRows(size_t startRow, size_t rowCount) const
        -> std::vector<std::tuple<nn::Tensor, std::array<int, 3>>>;
    [[nodiscard]] auto rowCount() const -> size_t;
    [[nodiscard]] auto filePath() const -> const std::string&;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class EEGLoader : public IMatLoader
{
   public:
    EEGLoader() = default;
    ~EEGLoader() override;

    auto open(const std::string& filePath) noexcept -> bool override; // flawfinder: ignore
    void close() noexcept override;

    auto readVariable(const std::string& name)
        -> std::unique_ptr<matvar_t, void (*)(matvar_t*)> override;

    std::optional<std::unique_ptr<matvar_t, void (*)(matvar_t*)>> readFirstNumericVariable()
        override;

    [[nodiscard]] auto filePath() const noexcept -> std::string override
    {
        return filePath_;
    }

   private:
    std::string filePath_;
    mat_t* matFile_ = nullptr;
};

// Convenience free function matching Audio loader pattern
//
// Data layout (assumed): the MAT variable is a 2D double matrix with shape (N_rows x 24579).
// - The first 24576 columns contain raw EEG samples. These are interpreted as 6 channels
//   × 4096 samples (total 24576). The loader returns a nn::Tensor with shape
//   (channels x samples_per_channel) == (6 x 4096).
// - The last three columns are labels: column 24577 = modality, 24578 = stimulus,
//   24579 = artifact (1-based description). In zero-based indexing used internally,
//   these are at indices 24576..24578.
//
// If your dataset stores samples interleaved across channels instead of contiguous blocks
// per channel, update `EEGLoader::loadEEGFromMat` to reshape accordingly (swap indexing
// when mapping the flat sample vector into the channels matrix).
//
// Returns: nn::Tensor of shape (channels x samples) and array of three ints: {modality, stimulus,
// artifact}
auto loadEEGFromMat(const std::string& filePath, size_t rowIndex)
    -> std::tuple<nn::Tensor, std::array<int, 3>>;

} // namespace nn::dataLoaders

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
#include <map>
#include <string>

#include "nn/dataLoaders/IMatLoader.hpp"
#include "nn/tensor/Tensor.hpp"

namespace nn::dataLoaders
{

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

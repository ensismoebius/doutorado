// EEGLoader.h
#pragma once

#include <Eigen/Dense>
#include <array>
#include <string>

#include "IMatLoader.h"

namespace nn::dataLoaders
{

class EEGLoader : public IMatLoader
{
   public:
    EEGLoader() = default;
    ~EEGLoader() override;

    auto open(const std::string& filePath) noexcept -> bool override;
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
// Returns: matrix of shape (channels x samples) and array of three ints: {modality, stimulus,
// artifact}
auto loadEEGFromMat(const std::string& filePath, size_t rowIndex)
    -> std::tuple<Eigen::MatrixXf, std::array<int, 3>>;

} // namespace nn::dataLoaders

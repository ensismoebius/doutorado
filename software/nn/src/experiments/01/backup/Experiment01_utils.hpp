#ifndef EXPERIMENT01_UTILS_HPP
#define EXPERIMENT01_UTILS_HPP

#include <Eigen/Dense>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "core/layers/Leaky.hpp"
#include "core/layers/Linear.hpp"
#include "core/layers/Module.hpp"
#include "core/layers/Sequential.hpp"
#include "core/tensor/Tensor.hpp"

/**
 * @brief Helper function to create a Sequential model.
 * @param list An initializer list of shared pointers to Module objects.
 * @return A unique pointer to the created Sequential model.
 */
inline auto make_sequential(std::initializer_list<std::shared_ptr<Module>> list)
    -> std::unique_ptr<Sequential>
{
    return std::make_unique<Sequential>(list);
}

class TraditionalAutoencoder : public Module
{
   private:
    std::unique_ptr<Sequential> encoder;
    std::unique_ptr<Sequential> decoder;

   public:
    TraditionalAutoencoder(int input_dim, int hidden_dim);

    auto forward(const Tensor& input) -> Tensor override;

    auto encode(const Tensor& input) -> Tensor;

    auto backward(const Tensor& grad_output) -> Tensor override;

    auto params() -> std::vector<Tensor*> override;
};

// Placeholder for Spiking Autoencoder
class SpikingAutoencoder : public Module
{
   private:
    std::unique_ptr<Sequential> encoder;
    std::unique_ptr<Sequential> decoder;

   public:
    SpikingAutoencoder(int input_dim, int hidden_dim);

    auto forward(const Tensor& input) -> Tensor override;

    auto encode(const Tensor& input) -> Tensor;

    auto backward(const Tensor& grad_output) -> Tensor override;

    auto params() -> std::vector<Tensor*> override;
};

void processSubject(const std::string& subjectPath, const std::string& subjectName,
                           const std::string& audioFilePath, const std::string& eegFilePath);

#endif // EXPERIMENT01_UTILS_HPP

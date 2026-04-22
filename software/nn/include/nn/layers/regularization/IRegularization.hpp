#ifndef I_REGULARIZATION_HPP
#define I_REGULARIZATION_HPP

#include <vector>

#include "nn/tensor/Tensor.hpp"

class IRegularization
{
   protected:
    float lambda_;

   public:
    explicit IRegularization(float lambda) : lambda_(lambda) {}
    virtual ~IRegularization() = default;

    virtual auto forward(const std::vector<nn::Tensor*>& params) -> nn::Tensor = 0;
    virtual void backward(const std::vector<nn::Tensor*>& params) = 0;
};

#endif // I_REGULARIZATION_HPP

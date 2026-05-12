#ifndef I_TRANSFORM_HPP
#define I_TRANSFORM_HPP

#include "tensor/Tensor.hpp"

namespace nn::transforms
{

class ITransform
{
   public:
    virtual ~ITransform() = default;
    virtual auto operator()(const nn::Tensor& x) const -> nn::Tensor = 0;
};

} // namespace nn::transforms

#endif // I_TRANSFORM_HPP
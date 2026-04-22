#ifndef COMPOSE_HPP
#define COMPOSE_HPP

#include <memory>
#include <numeric>
#include <vector>

#include "nn/utility/ITransform.hpp"

namespace nn::transforms
{

class Compose final : public ITransform
{
    std::vector<std::shared_ptr<ITransform>> steps_;

   public:
    explicit Compose(std::vector<std::shared_ptr<ITransform>> steps) : steps_(std::move(steps)) {}

    auto operator()(const nn::Tensor& x) const -> nn::Tensor override
    {
        return std::accumulate(steps_.cbegin(),
            steps_.cend(),
            x,
            [](const nn::Tensor& acc, const std::shared_ptr<ITransform>& t) { return (*t)(acc); });
    }
};

} // namespace nn::transforms

#endif // COMPOSE_HPP
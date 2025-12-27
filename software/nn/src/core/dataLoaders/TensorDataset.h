#pragma once

#include <algorithm>
#include <iterator>
#include <span>
#include <vector>

#include "Dataset.h"

class TensorDataset : public Dataset
{
   public:
    TensorDataset() = default;
    TensorDataset(nn::Tensor inputs, nn::Tensor targets)
        : inputs_(std::move(inputs)), targets_(std::move(targets))
    {
    }

    [[nodiscard]] auto get_item(std::size_t idx) const -> Batch override
    {
        std::vector<int> indices{static_cast<int>(idx)};
        nn::Tensor in = inputs_.slice(std::span<const int>(indices));
        nn::Tensor tg = targets_.slice(std::span<const int>(indices));
        return {.inputs = in, .targets = tg};
    }

    auto collate(const std::vector<std::size_t>& indices) const -> Batch override
    {
        std::vector<int> idxs;
        idxs.reserve(indices.size());
        std::transform(indices.begin(),
                       indices.end(),
                       std::back_inserter(idxs),
                       [](std::size_t i) { return static_cast<int>(i); });
        return {.inputs = inputs_.slice(std::span<const int>(idxs)),
                .targets = targets_.slice(std::span<const int>(idxs))};
    }

    [[nodiscard]] auto size() const -> std::size_t override
    {
        return static_cast<std::size_t>(inputs_.get_shape()[0]);
    }

   private:
    nn::Tensor inputs_;
    nn::Tensor targets_;

   protected:
    void set_tensors(nn::Tensor inputs, nn::Tensor targets)
    {
        inputs_ = std::move(inputs);
        targets_ = std::move(targets);
    }
};

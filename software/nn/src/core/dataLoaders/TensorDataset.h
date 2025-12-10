#pragma once

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
        nn::Tensor in = inputs_.slice(indices);
        nn::Tensor tg = targets_.slice(indices);
        return {.inputs = in, .targets = tg};
    }

    auto collate(const std::vector<std::size_t>& indices) const -> Batch override
    {
        std::vector<int> idxs;
        idxs.reserve(indices.size());
        for (auto i : indices)
        {
            idxs.push_back(static_cast<int>(i));
        }
        return {.inputs = inputs_.slice(idxs), .targets = targets_.slice(idxs)};
    }

    [[nodiscard]] auto size() const -> std::size_t override
    {
        return static_cast<std::size_t>(inputs_.get_shape()[0]);
    }

   private:
    nn::Tensor inputs_;
    nn::Tensor targets_;
};

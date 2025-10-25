#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "../tensor/Tensor.hpp"
#include "util/batching.hpp"

// Abstract dataset interface similar to PyTorch's Dataset
class Dataset {
 public:
  [[nodiscard]] virtual auto get_item(std::size_t idx) const -> Batch = 0;
  [[nodiscard]] virtual auto collate(
      const std::vector<std::size_t>& indices) const -> Batch {
    // Default collate: call get_item for each index and stack rows.
    if (indices.empty()) {
      return Batch{};
    }

    Batch first = get_item(indices[0]);
    const Eigen::Index cols_in = first.inputs.data.cols();
    const Eigen::Index cols_tg = first.targets.data.cols();

    Eigen::MatrixXf inputs_mat(static_cast<int>(indices.size()),
                               static_cast<int>(cols_in));
    Eigen::MatrixXf targets_mat(static_cast<int>(indices.size()),
                                static_cast<int>(cols_tg));

    for (std::size_t i = 0; i < indices.size(); ++i) {
      Batch b = get_item(indices[i]);
      inputs_mat.row(static_cast<int>(i)) = b.inputs.data.row(0);
      targets_mat.row(static_cast<int>(i)) = b.targets.data.row(0);
    }

    return {.inputs = Tensor(inputs_mat), .targets = Tensor(targets_mat)};
  }

  [[nodiscard]] virtual auto size() const -> std::size_t = 0;
  virtual ~Dataset() = default;
};

// Concrete dataset that wraps full input/target tensors
class TensorDataset : public Dataset {
 public:
  TensorDataset() = default;
  TensorDataset(Tensor inputs, Tensor targets)
      : inputs_(std::move(inputs)), targets_(std::move(targets)) {}

  [[nodiscard]] auto get_item(std::size_t idx) const -> Batch override {
    std::vector<int> indices{static_cast<int>(idx)};
    Tensor in = inputs_.slice(indices);
    Tensor tg = targets_.slice(indices);
    return {.inputs = in, .targets = tg};
  }

  auto collate(const std::vector<std::size_t>& indices) const
      -> Batch override {
    // Efficient slice-based collate
    std::vector<int> idxs;
    idxs.reserve(indices.size());
    for (auto i : indices) {
      idxs.push_back(static_cast<int>(i));
    }
    return {.inputs = inputs_.slice(idxs), .targets = targets_.slice(idxs)};
  }

  auto size() const -> std::size_t override {
    return static_cast<std::size_t>(inputs_.get_shape()[0]);
  }

 private:
  Tensor inputs_;
  Tensor targets_;
};

class DataLoader {
 public:
  // dataset: shared_ptr to a Dataset; batch_size: size_t; optional seed for
  // deterministic shuffle
  DataLoader(std::shared_ptr<Dataset> dataset, std::size_t batch_size,
             bool shuffle = true,
             std::optional<unsigned int> seed = std::nullopt);

  class Iterator {
   public:
    Iterator(DataLoader& loader, std::size_t current_batch);

    auto operator*() const -> Batch;
    auto operator++() -> Iterator&;
    auto operator!=(const Iterator& other) const -> bool;

   private:
    DataLoader& loader_;
    std::size_t current_batch_;
  };

  [[nodiscard]] auto begin() -> Iterator;
  [[nodiscard]] auto end() -> Iterator;

 private:
  std::shared_ptr<Dataset> dataset_;
  std::size_t batch_size_;
  bool shuffle_;
  std::optional<unsigned int> seed_;
  std::size_t num_batches_;
  std::vector<std::size_t> indices_;
  // epoch counter used when seed_ is present to vary shuffle between epochs
  mutable std::size_t epoch_ = 0;
};

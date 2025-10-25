#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "Dataset.h"

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

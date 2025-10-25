#pragma once

#include <vector>

#include "../tensor/Tensor.hpp"
#include "util/batching.hpp"

struct Dataset {
  Tensor inputs;
  Tensor targets;
};

class DataLoader {
 public:
  DataLoader(Dataset dataset, int batch_size, bool shuffle = true);

  class Iterator {
   public:
    Iterator(const DataLoader& loader, long current_batch);

    auto operator*() const -> Batch;
    auto operator++() -> Iterator&;
    auto operator!=(const Iterator& other) const -> bool;

   private:
    const DataLoader& loader_;
    long current_batch_;
  };

  [[nodiscard]] auto begin() const -> Iterator;
  [[nodiscard]] auto end() const -> Iterator;

 private:
  Dataset dataset_;
  int batch_size_;
  bool shuffle_;
  long num_batches_;
  std::vector<int> indices_;
};

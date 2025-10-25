// DataLoader test file
#include "dataLoaders/DataLoader.h"
// #include "dataLoaders/MatFile.h"  // unused in this simple test

auto main() -> int {
  // Create an empty TensorDataset for test purposes
  Tensor inputs(0, 0);
  Tensor targets(0, 0);
  auto dataset = std::make_shared<TensorDataset>(inputs, targets);
  DataLoader loader(dataset, static_cast<std::size_t>(32), true);
  for (const auto& batch : loader) {
    // Process batch
  }
  return 0;
}
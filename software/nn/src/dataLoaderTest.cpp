// DataLoader test file
#include "dataLoaders/DataLoader.h"
#include "dataLoaders/MatFile.h"

using namespace matio;

auto main() -> int {
  Dataset dataset;
  DataLoader loader(dataset, 32L);  // Using long for batch size
  for (const auto& batch : loader) {
    // Process batch
  }
  return 0;
}
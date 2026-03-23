#include <iostream>
#include <thread>
#include <chrono>
#include "nn/dataLoaders/BatchPrefetcher.hpp"
#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/dataLoaders/TensorDataset.hpp"

auto make_sequential_tensor(std::size_t rows, std::size_t cols) -> nn::Tensor
{
    nn::Tensor t(rows, cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            t.at(i, j) = static_cast<float>((i * cols) + j);
        }
    }
    return t;
}

int main() {
    auto inputs = make_sequential_tensor(1000, 128);
    auto targets = make_sequential_tensor(1000, 16);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);
    DataLoader loader(dataset, 10, false);

    // 100 batches total. Lookahead of 10.
    BatchPrefetcher prefetcher(loader, 100, 10, false, "", 0);

    // Give the producer a head start so it fills the queue
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    int batches_read = 0;
    while (auto batch = prefetcher.next()) {
        batches_read++;
        // Simulate training step time to let producer keep queue full
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto d = prefetcher.diagnostics();
    std::cout << "Done. Batches read: " << batches_read << "\n";
    std::cout << "Fast path hits: " << d.fast_path_hits << "\n";
    std::cout << "Slow path hits: " << d.slow_path_hits << "\n";
    std::cout << "Push successes: " << d.push_successes << "\n";
    std::cout << "Push retries: " << d.push_retries << "\n";
    
    if (d.fast_path_hits > d.slow_path_hits) {
        std::cout << "SUCCESS: Fast path dominates!\n";
        return 0;
    } else {
        std::cout << "FAIL: Slow path dominates.\n";
        return 1;
    }
}

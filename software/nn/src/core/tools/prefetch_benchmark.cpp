#include <chrono>
#include <iostream>
#include <memory>

#include "nn/dataLoaders/BatchPrefetcher.hpp"
#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/dataLoaders/DataLoaderBatchSource.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/utility/batching.hpp"

using namespace std::chrono;

// A simple synthetic dataset that returns random tensors of fixed shape.
class FakeDataset : public Dataset
{
   public:
    FakeDataset(std::size_t n, int in_cols, int tg_cols)
        : n_(n), in_cols_(in_cols), tg_cols_(tg_cols)
    {
    }

    auto get_item(std::size_t idx) const -> Batch override
    {
        (void) idx;
        Batch b;
        b.inputs = nn::Tensor(1, in_cols_);
        b.targets = nn::Tensor(1, tg_cols_);
        // Fill with deterministic data to avoid uninitialized memory effects
        for (nn::Index i = 0; i < b.inputs.rows(); ++i)
            for (nn::Index j = 0; j < b.inputs.cols(); ++j)
                b.inputs(i, j) = static_cast<float>((i + j) % 127) / 127.0F;
        for (nn::Index i = 0; i < b.targets.rows(); ++i)
            for (nn::Index j = 0; j < b.targets.cols(); ++j)
                b.targets(i, j) = static_cast<float>((i + j + 3) % 97) / 97.0F;
        return b;
    }

    void collate_into(const std::vector<std::size_t>& indices, Batch& batch) const override
    {
        // Use default implementation for simplicity
        Dataset::collate_into(indices, batch);
    }

    auto size() const -> std::size_t override
    {
        return n_;
    }

   private:
    std::size_t n_;
    int in_cols_;
    int tg_cols_;
};

static void run_test(bool use_move, std::size_t dataset_size, std::size_t batch_size)
{
    auto ds = std::make_shared<FakeDataset>(dataset_size, 256, 10);
    DataLoader loader(ds, batch_size, /*do_shuffle=*/false);

    auto it = loader.begin();
    auto end = loader.end();

    std::size_t batches = 0;
    std::size_t samples = 0;

    auto t0 = high_resolution_clock::now();
    while (it != end)
    {
        if (use_move)
        {
            Batch b = it.move_batch();
            (void) b;
        }
        else
        {
            Batch b = *it; // copy from iterator's internal buffer
            (void) b;
        }
        ++it;
        ++batches;
        samples += batch_size;
    }
    auto t1 = high_resolution_clock::now();
    duration<double> s = t1 - t0;

    double secs = s.count();
    std::cout << (use_move ? "Move" : "Copy") << " mode: " << batches << " batches in " << secs
              << "s -> " << (batches / secs) << " batches/s, " << (samples / secs)
              << " samples/s\n";
}

static void run_prefetcher(std::size_t dataset_size, std::size_t batch_size)
{
    auto ds = std::make_shared<FakeDataset>(dataset_size, 256, 10);
    DataLoader loader(ds, batch_size, /*do_shuffle=*/false);

    const std::size_t max_batches = (dataset_size + batch_size - 1) / batch_size;
    // lookahead 4
    BatchPrefetcher prefetcher(std::make_unique<DataLoaderBatchSource>(loader), max_batches, 4);

    std::size_t batches = 0;
    std::size_t samples = 0;

    auto t0 = high_resolution_clock::now();
    while (true)
    {
        auto ob = prefetcher.next();
        if (!ob) break;
        Batch b = std::move(*ob);
        (void) b;
        ++batches;
        samples += batch_size;
    }
    auto t1 = high_resolution_clock::now();
    duration<double> s = t1 - t0;
    double secs = s.count();
    std::cout << "Prefetcher (SPSC) mode: " << batches << " batches in " << secs << "s -> "
              << (batches / secs) << " batches/s, " << (samples / secs) << " samples/s\n";
}

int main()
{
    const std::size_t dataset_size = 10000; // samples
    const std::size_t batch_size = 32;

    std::cout << "Prefetch microbenchmark (synthetic dataset)\n";
    std::cout << "Dataset size: " << dataset_size << " samples, batch_size=" << batch_size << "\n";

    run_test(false, dataset_size, batch_size);
    run_test(true, dataset_size, batch_size);
    run_prefetcher(dataset_size, batch_size);

    return 0;
}

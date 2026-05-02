#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "nn/layers/dense/Linear.hpp"
#include "nn/tensor/opencl/OpenCLContext.hpp"
#include "nn/tensor/opencl/OpenCLTensorBackend.hpp"
#include "nn/tensor/xtensor/XTensorBackend.hpp"

namespace
{
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

struct BenchResult
{
    std::string backend;
    std::string operation;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t inner = 0;
    int iterations = 0;
    double total_ms = 0.0;
    double per_iter_ms = 0.0;
};

auto env_to_int(const char* name, int fallback) -> int
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return fallback;
    }
    return std::max(1, std::atoi(value));
}

template <typename Fn>
auto measure(const std::string& backend,
    const std::string& operation,
    std::size_t rows,
    std::size_t cols,
    std::size_t inner,
    int iterations,
    Fn&& fn) -> BenchResult
{
    const auto start = Clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        fn();
    }
    const double total_ms = Ms(Clock::now() - start).count();
    return BenchResult{
        .backend = backend,
        .operation = operation,
        .rows = rows,
        .cols = cols,
        .inner = inner,
        .iterations = iterations,
        .total_ms = total_ms,
        .per_iter_ms = total_ms / static_cast<double>(iterations),
    };
}

template <typename Backend>
auto benchmark_backend(const std::string& backend_name, int iterations) -> std::vector<BenchResult>
{
    using Tensor = Backend;
    std::vector<BenchResult> results;
    std::mt19937 rng(1337);

    Tensor lhs = Tensor::random(256, 256, rng);
    Tensor rhs = Tensor::random(256, 256, rng); // Context line
    Tensor input = Tensor::random(1024, 256, rng);
    Tensor weight = Tensor::random(512, 256, rng);
    Tensor bias = Tensor::ones(512, 1);
    Tensor grad_output = Tensor::random(1024, 512, rng);

    results.push_back(measure(backend_name,
        "random_1024x1024",
        1024,
        1024,
        0,
        iterations,
        [&rng]() mutable
        {
            auto tensor = Tensor::random(1024, 1024, rng);
            volatile float sink = tensor.at(0, 0);
            (void) sink;
        }));

    results.push_back(measure(backend_name,
        "fill_zero_1024x1024",
        1024,
        1024,
        0,
        iterations,
        [&]()
        {
            auto tensor = Tensor::zeros(1024, 1024);
            volatile float sink = tensor.at(0, 0);
            (void) sink;
        }));

    results.push_back(measure(backend_name,
        "matmul_256",
        256,
        256,
        256,
        iterations,
        [&]()
        {
            auto product = lhs.matmul(rhs);
            volatile float sink = product.at(0, 0);
            (void) sink;
        }));

    results.push_back(measure(backend_name,
        "linear_chain_1024x256x512",
        1024,
        512,
        256,
        iterations,
        [&]()
        {
            auto output = input.matmul_transposed(weight);
            output.add_col_vector_to_rows_inplace(bias);
            volatile float sink = output.at(0, 0);
            (void) sink;
        }));

    results.push_back(measure(backend_name,
        "linear_backward_chain_1024x256x512",
        1024,
        512,
        256,
        iterations,
        [&]()
        {
            auto grad_t = grad_output.transpose();
            auto grad_weight = grad_t.matmul(input);
            auto grad_bias = grad_t.rowwise_sum();
            auto grad_input = grad_output.matmul(weight);
            volatile float sink = grad_weight.at(0, 0) + grad_bias.at(0, 0) + grad_input.at(0, 0);
            (void) sink;
        }));

    if constexpr (std::is_same_v<Backend, nn::Backend>)
    {
        using LayerTensor = nn::TensorImpl<Backend>;
        LinearImpl<Backend> linear(256, 512);
        linear.weight = nn::Tensor::rand(512, 256, rng);
        linear.bias = nn::Tensor::ones(512, 1);
        LayerTensor layer_input = LayerTensor::rand(1024, 256, rng);

        results.push_back(measure(backend_name,
            "linear_layer_forward_1024x256x512",
            1024,
            512,
            256,
            iterations,
            [&]()
            {
                auto output = linear.forward(layer_input, false);
                volatile float sink = output.at(0, 0);
                (void) sink;
            }));
    }

    return results;
}

void print_results_csv(const std::vector<BenchResult>& results)
{
    std::cout << "backend,operation,rows,cols,inner,iterations,total_ms,per_iter_ms\n";
    std::cout << std::fixed << std::setprecision(3);
    for (const auto& result : results)
    {
        std::cout << result.backend << ',' << result.operation << ',' << result.rows << ','
                  << result.cols << ',' << result.inner << ',' << result.iterations << ','
                  << result.total_ms << ',' << result.per_iter_ms << '\n';
    }
}
} // namespace

int main()
{
    const int iterations = env_to_int("NN_TENSOR_BENCH_ITERS", 5);
    std::vector<BenchResult> results = benchmark_backend<nn::XTensorBackend>("xtensor", iterations);

    if (nn::opencl::OpenCLContext::instance().is_available())
    {
        auto opencl_results = benchmark_backend<nn::OpenCLTensorBackend>("opencl", iterations);
        results.insert(results.end(), opencl_results.begin(), opencl_results.end());
    }

    print_results_csv(results);
    return 0;
}
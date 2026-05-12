#include <chrono>
#include <iomanip>
#include <iostream>

#include "layers/activations/FastActivations.hpp"
#include "tensor/Tensor.hpp"

using Tensor = nn::Tensor;
using Clock = std::chrono::high_resolution_clock;

struct Timing
{
    const char* name;
    double ms;
};

int main()
{
    // LSTM-bench config: B=1, D=128 (audio features), H=32 (hidden)
    const int B = 1;
    const int D = 128;
    const int H = 32;
    const int T = 256; // time steps
    const int reps = 100;

    // Matrices
    Tensor W(4 * H, D);   // (128, 128)
    Tensor U(4 * H, H);   // (128, 32)
    Tensor x_t(B, D);     // (1, 128)
    Tensor h(B, H);       // (1, 32)
    Tensor pre(B, 4 * H); // (1, 128) — stacked 4 gates


    // Initialize with random
    for (nn::Index i = 0; i < W.rows(); ++i)
        for (nn::Index j = 0; j < W.cols(); ++j) W.at(i, j) = 0.1f;
    for (nn::Index i = 0; i < U.rows(); ++i)
        for (nn::Index j = 0; j < U.cols(); ++j) U.at(i, j) = 0.05f;
    for (nn::Index i = 0; i < x_t.rows(); ++i)
        for (nn::Index j = 0; j < x_t.cols(); ++j) x_t.at(i, j) = 0.02f;
    for (nn::Index i = 0; i < h.rows(); ++i)
        for (nn::Index j = 0; j < h.cols(); ++j) h.at(i, j) = 0.01f;

    std::vector<Timing> results;

    // Benchmark 1: x @ W^T (main matmul cost)
    {
        auto start = Clock::now();
        for (int r = 0; r < reps; ++r)
        {
            auto result = x_t.matmul_transposed(W);
            (void) result;
        }
        auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        results.push_back({"x_t @ W^T matmul", elapsed / reps});
    }

    // Benchmark 2: h @ U^T (secondary matmul)
    {
        auto start = Clock::now();
        for (int r = 0; r < reps; ++r)
        {
            auto result = h.matmul_transposed(U);
            (void) result;
        }
        auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        results.push_back({"h @ U^T matmul", elapsed / reps});
    }

    // Benchmark 3: sigmoid_fast on (1, 32)
    {
        Tensor gate(B, H);
        auto start = Clock::now();
        for (int r = 0; r < reps * T; ++r)
        { // T × more reps since gate ops are fast
            gate = nn::activations::sigmoid_fast_tensor(gate);
        }
        auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        results.push_back({"sigmoid_fast_tensor (1,32)", elapsed / (reps * T)});
    }

    // Benchmark 4: tanh_fast on (1, 32)
    {
        Tensor gate(B, H);
        auto start = Clock::now();
        for (int r = 0; r < reps * T; ++r)
        {
            gate = nn::activations::tanh_fast_tensor(gate);
        }
        auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        results.push_back({"tanh_fast_tensor (1,32)", elapsed / (reps * T)});
    }

    // Benchmark 5a: 4 gates via block() + activation (old way)
    {
        auto start = Clock::now();
        for (int r = 0; r < reps * T; ++r)
        {
            auto i_g = nn::activations::sigmoid_fast_tensor(pre.block(0, 0, B, H));
            auto f_g = nn::activations::sigmoid_fast_tensor(pre.block(0, 1 * H, B, H));
            auto o_g = nn::activations::sigmoid_fast_tensor(pre.block(0, 2 * H, B, H));
            auto g_g = nn::activations::tanh_fast_tensor(pre.block(0, 3 * H, B, H));
            (void) i_g;
            (void) f_g;
            (void) o_g;
            (void) g_g;
        }
        auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        results.push_back({"4_gates block()+fast", elapsed / (reps * T)});
    }

    // Benchmark 5b: 4 gates via fused block+activation (new way)
    {
        auto start = Clock::now();
        for (int r = 0; r < reps * T; ++r)
        {
            auto i_g = nn::activations::sigmoid_fast_block(pre, 0 * H, H);
            auto f_g = nn::activations::sigmoid_fast_block(pre, 1 * H, H);
            auto o_g = nn::activations::sigmoid_fast_block(pre, 2 * H, H);
            auto g_g = nn::activations::tanh_fast_block(pre, 3 * H, H);
            (void) i_g;
            (void) f_g;
            (void) o_g;
            (void) g_g;
        }
        auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        results.push_back({"4_gates fused_block", elapsed / (reps * T)});
    }

    // Benchmark 6a: full timestep — old (block + fast)
    {
        Tensor c(B, H);
        auto start = Clock::now();
        for (int r = 0; r < reps; ++r)
        {
            auto pre_sum = x_t.matmul_transposed(W).add(h.matmul_transposed(U));
            auto i_g = nn::activations::sigmoid_fast_tensor(pre_sum.block(0, 0, B, H));
            auto f_g = nn::activations::sigmoid_fast_tensor(pre_sum.block(0, 1 * H, B, H));
            auto o_g = nn::activations::sigmoid_fast_tensor(pre_sum.block(0, 2 * H, B, H));
            auto g_g = nn::activations::tanh_fast_tensor(pre_sum.block(0, 3 * H, B, H));
            auto c_new = (f_g * c).add(i_g * g_g);
            auto tc = nn::activations::tanh_fast_tensor(c_new);
            auto h_new = o_g * tc;
            (void) h_new;
        }
        auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        results.push_back({"full_timestep_old", elapsed / reps});
    }

    // Benchmark 6b: full timestep — new (fused block+activation)
    {
        Tensor c(B, H);
        auto start = Clock::now();
        for (int r = 0; r < reps; ++r)
        {
            auto pre_sum = x_t.matmul_transposed(W).add(h.matmul_transposed(U));
            auto i_g = nn::activations::sigmoid_fast_block(pre_sum, 0 * H, H);
            auto f_g = nn::activations::sigmoid_fast_block(pre_sum, 1 * H, H);
            auto o_g = nn::activations::sigmoid_fast_block(pre_sum, 2 * H, H);
            auto g_g = nn::activations::tanh_fast_block(pre_sum, 3 * H, H);
            auto c_new = (f_g * c).add(i_g * g_g);
            auto tc = nn::activations::tanh_fast_tensor(c_new);
            auto h_new = o_g * tc;
            (void) h_new;
        }
        auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        results.push_back({"full_timestep_fused", elapsed / reps});
    }

    // Report
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║ LSTM Microbench (B=1, D=128, H=32, T=256)             ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    std::cout << std::left << std::setw(30) << "Operation" << std::setw(15) << "Time (ms)"
              << "% of Full Step\n";
    std::cout << std::string(55, '-') << "\n";

    double full_step_time = 0;
    for (const auto& r : results)
    {
        if (std::string(r.name) == "full_timestep_old")
        {
            full_step_time = r.ms;
            break;
        }
    }

    for (const auto& r : results)
    {
        double pct = full_step_time > 0 ? (r.ms / full_step_time) * 100.0 : 0;
        std::cout << std::left << std::setw(30) << r.name << std::fixed << std::setprecision(4)
                  << std::setw(15) << r.ms << pct << "%\n";
    }

    std::cout << "\n✓ Bottleneck identification complete.\n";
    return 0;
}

/**
 * @file EigenParallel.hpp
 * @brief Small helper to align Eigen and OpenMP thread configuration.
 *
 * This is best-effort runtime tuning. Deterministic experiments should still pin
 * thread counts explicitly (e.g., set `OMP_NUM_THREADS=1`).
 */

#ifndef EIGEN_PARALLEL_HPP
#define EIGEN_PARALLEL_HPP

#include <Eigen/Dense>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace util
{
/**
 * @brief Initialize Eigen and OpenMP thread settings.
 *
 * Behavior:
 * - If `numThreads > 0` it will be used as both Eigen and OpenMP thread count.
 * - If `numThreads == 0` we prefer `std::thread::hardware_concurrency()` as a
 *   reliable upper bound. If that returns 0, fall back to Eigen::nbThreads().
 *
 * This helps avoid cases where Eigen reports a too-small default (e.g. 1 or
 * half the logical cores) which would under-utilize the machine.
 */
inline auto initializeEigenParallel(int numThreads = 0) -> void
{
    if (numThreads <= 0)
    {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0)
        {
            // Hardware concurrency unknown: fall back to Eigen's current value
            numThreads = Eigen::nbThreads();
        }
        else
        {
            numThreads = static_cast<int>(hc);
        }
    }

#ifdef _OPENMP
    // Inform OpenMP runtime about the desired thread count as well.
    omp_set_num_threads(numThreads);
#if _OPENMP >= 200805
    // Disable nested parallelism by default to avoid oversubscription (OpenMP 3.0+)
    omp_set_max_active_levels(1);
#else
    // Disable nested parallelism by default to avoid oversubscription
    omp_set_nested(0);
#endif
#endif

    // Set Eigen's internal threading to match.
    Eigen::setNbThreads(numThreads);
}

} // namespace util

#endif // EIGEN_PARALLEL_HPP

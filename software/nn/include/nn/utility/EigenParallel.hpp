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
    // Disable nested parallelism by default to avoid oversubscription.
    omp_set_nested(0);
#endif

    // Set Eigen's internal threading to match.
    Eigen::setNbThreads(numThreads);
}

} // namespace util

#endif // EIGEN_PARALLEL_HPP

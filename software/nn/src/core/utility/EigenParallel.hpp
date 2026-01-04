#ifndef EIGEN_PARALLEL_HPP
#define EIGEN_PARALLEL_HPP

#include "core/tensor/EigenTensorBackend.hpp"

namespace util
{
/**
 * @brief Initialize Eigen's parallel execution settings
 *
 * @param numThreads Number of threads to use. If 0, uses maximum available threads
 */
inline auto initializeEigenParallel(int numThreads = 0) -> void
{
    if (numThreads <= 0)
    {
        // Get the maximum number of threads available
        using namespace Eigen;
        numThreads = nbThreads();
    }

    // Set the number of threads Eigen will use
    using namespace Eigen;
    setNbThreads(numThreads);
}
} // namespace util

#endif // EIGEN_PARALLEL_HPP

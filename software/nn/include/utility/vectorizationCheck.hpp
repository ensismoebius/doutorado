#ifndef VECTORIZATION_SUPPORT
#define VECTORIZATION_SUPPORT

/**
 * @file vectorizationCheck.hpp
 * @brief Small diagnostics helper: prints CPU SIMD/vectorization capabilities.
 *
 * Why this exists:
 * - xtensor performance depends on SIMD and parallelization flags.
 * - The demos print this at startup so you can sanity-check whether you are
 *   getting acceleration on the current machine/build.
 */
void printVectorizationSupport();

#endif
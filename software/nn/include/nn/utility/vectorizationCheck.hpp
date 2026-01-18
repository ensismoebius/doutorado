#ifndef VECTORIZATION_SUPPORT
#define VECTORIZATION_SUPPORT

/**
 * @file vectorizationCheck.hpp
 * @brief Small diagnostics helper: prints CPU SIMD/vectorization capabilities.
 *
 * Why this exists:
 * - Eigen performance depends heavily on SIMD support and compile flags.
 * - The demos print this at startup so you can sanity-check whether you are
 *   getting AVX/SSE/NEON acceleration on the current machine/build.
 */
void printVectorizationSupport();

#endif
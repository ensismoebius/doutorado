/**
 * @file testing.hpp
 * @brief Small shared testing constants for deterministic tests.
 */

#ifndef NN_TESTING_HPP
#define NN_TESTING_HPP

namespace nn
{
namespace testing
{
// Global seed used across tests and deterministic demos when requested.
inline constexpr unsigned kSeed = 42U;

// Default floating-point tolerance for tests.
inline constexpr float kTol = 1e-6F;

} // namespace testing
} // namespace nn

#endif // NN_TESTING_HPP

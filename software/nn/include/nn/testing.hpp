/**
 * @file testing.hpp
 * @brief Small shared testing constants for deterministic tests.
 */

#pragma once

namespace nn
{
namespace testing
{
// Global seed used across tests and deterministic demos when requested.
inline constexpr unsigned SEED = 42U;

// Default floating-point tolerance for tests.
inline constexpr float TOL = 1e-6F;

} // namespace testing
} // namespace nn

#ifndef NN_UTILITY_SEEDDERIVATION_HPP
#define NN_UTILITY_SEEDDERIVATION_HPP

/**
 * @file include/utility/SeedDerivation.hpp
 * @brief One run seed -> one independent 32-bit seed per random stream (weight tensor,
 *        layer, sampler) of that run.
 *
 * Why not `base + offset`: two runs seeded 42 and 43 would then share streams. Run 42's
 * second tensor (42 + 1) would be run 43's first tensor (43 + 0), so the "two seeds" would
 * start from partly identical weights. std::seed_seq mixes (base, stream) through the
 * standard's fully specified algorithm, so every (base, stream) pair gets its own seed. The
 * same pair gives the same seed on every platform and compiler.
 */

#include <array>
#include <cstdint>
#include <random>

namespace nn::utility
{

/// The seed of stream `stream` of a run seeded `base`. Deterministic and portable; distinct
/// (base, stream) pairs give unrelated seeds (seed_sequence_gtest: SeedDerivation.*).
inline auto derive_seed(std::uint32_t base, std::uint32_t stream) -> std::uint32_t
{
    std::seed_seq seq{base, stream};
    std::array<std::uint32_t, 1> out{};
    seq.generate(out.begin(), out.end());
    return out[0];
}

} // namespace nn::utility

#endif // NN_UTILITY_SEEDDERIVATION_HPP

#ifndef RANDOM_INDEX_CROP_HPP
#define RANDOM_INDEX_CROP_HPP

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

namespace nn::transforms
{

// The discrete counterpart of RandomCrop (see RandomCrop.hpp), for datasets that pre-slice
// every candidate window up front instead of cropping lazily from a continuous signal per
// access -- this framework's meeting01 loaders do this deliberately, so every window has a
// stable window_id/source_window_index for leakage-safe splitting. Where RandomCrop picks one
// random offset into a continuous signal, RandomIndexCrop picks a random ORDER over an already
// -materialized set of candidate indices: callers drain the returned vector front-to-back (as
// stratified_window_cap does) to consume candidates in a random, reproducible-per-seed order
// instead of always draining index 0 first.
//
// Not an ITransform: it permutes indices, not tensor values, so it doesn't fit
// operator()(Tensor) -> Tensor. Stateful like RandomCrop -- the instance owns its RNG and
// advances it on every call, so the SAME instance called once per recording (as
// stratified_window_cap does) produces one continuous, seed-reproducible draw sequence across
// all of them, not an independent reseed per recording.
class RandomIndexCrop
{
    mutable std::mt19937 rng_;

   public:
    explicit RandomIndexCrop(unsigned int seed) : rng_(seed) {}

    auto operator()(std::vector<std::size_t> candidates) const -> std::vector<std::size_t>
    {
        std::shuffle(candidates.begin(), candidates.end(), rng_);
        return candidates;
    }
};

} // namespace nn::transforms

#endif // RANDOM_INDEX_CROP_HPP

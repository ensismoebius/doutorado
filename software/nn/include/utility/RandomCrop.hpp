#ifndef RANDOM_CROP_HPP
#define RANDOM_CROP_HPP

#include <random>
#include <stdexcept>

#include "utility/ITransform.hpp"

namespace nn::transforms
{

// torchvision.transforms.RandomCrop, for a 1-D signal stored as an (N, 1) column tensor:
// given a signal with N >= crop_size samples, returns a (crop_size, 1) window starting at a
// uniformly random valid offset in [0, N - crop_size]. The instance owns its RNG and advances
// it on every call (never re-seeded per call) -- same stateful-callable semantics as
// torchvision's RandomCrop, so reproducibility comes from the seed passed once at construction,
// not from anything the caller does per call.
//
// N == crop_size collapses to the only valid offset (0) and consumes no randomness, matching
// torchvision's behaviour when crop size equals image size. See RandomIndexCrop.hpp for the
// discrete counterpart used where candidate windows are already materialized (not cropped from
// a continuous signal).
class RandomCrop final : public ITransform
{
    nn::Index crop_size_;
    mutable std::mt19937 rng_;

   public:
    RandomCrop(nn::Index crop_size, unsigned int seed) : crop_size_(crop_size), rng_(seed)
    {
        if (crop_size_ == 0) throw std::invalid_argument("RandomCrop: crop_size must be > 0");
    }

    auto operator()(const nn::Tensor& x) const -> nn::Tensor override
    {
        if (x.cols() != 1)
            throw std::invalid_argument("RandomCrop: expected a (N, 1) column-vector signal");
        if (x.rows() < crop_size_)
            throw std::invalid_argument(
                "RandomCrop: signal has fewer rows than crop_size -- no valid crop exists");

        if (x.rows() == crop_size_) return x;

        std::uniform_int_distribution<nn::Index> pick(0, x.rows() - crop_size_);
        const nn::Index offset = pick(rng_);
        return x.block(offset, 0, crop_size_, 1);
    }
};

} // namespace nn::transforms

#endif // RANDOM_CROP_HPP

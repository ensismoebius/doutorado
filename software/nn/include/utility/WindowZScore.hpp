#ifndef WINDOW_ZSCORE_HPP
#define WINDOW_ZSCORE_HPP

#include "utility/ITransform.hpp"
#include "utility/SignalPreprocessing.hpp"

namespace nn::transforms
{

// Whole-tensor z-score, wrapped as a composable ITransform around
// nn::utility::zscore_inplace (one canonical numeric implementation, reused here rather than
// re-derived). Normalizes every element of the tensor against ONE mean/std computed over all
// of it -- the right convention for a single-channel window stored as an (N, 1) column vector
// (FSDD/AudioMNIST/EEG/MIT-BIH loaders all use this shape).
//
// Do not confuse with the other two normalizers in this framework, which use DIFFERENT
// conventions for DIFFERENT tensor shapes:
//   - EEGWindowZScore  -- per-ROW z-score across columns; expects (channels, time).
//   - AudioMeanStdNormalize -- per-COLUMN z-score, fitted across a whole batch/dataset (not
//     per-sample); expects (samples, features).
// Applying either of those to an (N, 1) window would z-score each 1-element row against
// itself (mean = the value, std = 0) -- a silent no-op that looks like it ran correctly.
class WindowZScore final : public ITransform
{
   public:
    auto operator()(const nn::Tensor& x) const -> nn::Tensor override
    {
        nn::Tensor out = x;
        nn::utility::zscore_inplace(out);
        return out;
    }
};

} // namespace nn::transforms

#endif // WINDOW_ZSCORE_HPP

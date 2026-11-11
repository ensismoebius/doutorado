#ifndef RESAMPLE_HPP
#define RESAMPLE_HPP

#include <vector>

namespace nn::utility
{

// Converts a signal sampled at from_hz to to_hz with a Hann-windowed sinc kernel.
//
// THE PROBLEM. A network reads a window as "256 numbers" and never sees the clock. Most
// EEGMMIDB files run at 160 Hz, three subjects' at 128 Hz: 256 of their samples last 2.0 s, not
// 1.6 s, and every frequency looks 0.8 times its real value. A 10 Hz rhythm at 128 Hz is, sample
// for sample, a 12.5 Hz rhythm at 160 Hz. Nothing fails; the data are simply not comparable.
//
//   128 Hz:  x---x---x---x---x      one sample every 7.8 ms
//   160 Hz:  x--x--x--x--x--x--x    one sample every 6.25 ms, each between two of the above
//
// Each output sample is read off the continuous signal the input samples define, with the
// sinc interpolation of the sampling theorem: y(t) = sum_k x[k] c sinc(c (t - k)) w(t - k),
// t in input samples, c = min(1, to_hz / from_hz). c < 1 lowers the cut-off to the new Nyquist
// so a downsample does not alias. The kernel spans half_width_zero_crossings zero crossings
// each side and is renormalized at each output sample (near the ends it is cut off by the
// recording), so a constant signal stays constant. Output sample 0 is input sample 0; the last
// output sample is the last whole one inside the input.
//
// from_hz == to_hz returns the input. Throws std::invalid_argument on a non-positive rate, an
// empty signal or half_width_zero_crossings < 1.
auto resample_bandlimited(const std::vector<float>& signal,
    double from_hz,
    double to_hz,
    int half_width_zero_crossings = 32) -> std::vector<float>;

} // namespace nn::utility

#endif // RESAMPLE_HPP

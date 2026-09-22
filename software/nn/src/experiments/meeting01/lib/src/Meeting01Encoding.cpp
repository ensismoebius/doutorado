#include "../include/Meeting01Encoding.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

namespace meeting01
{

auto encode_sample(
    const Tensor& sample, const std::string& encoding, std::uint32_t seed, int time_steps) -> Tensor
{
    // Layout contract (changed 2026-09-22). A window arrives as (window_size, 1) and
    // leaves as time-major (T, F): row = simulation step, column = one window sample,
    // F = window_size. Every window sample is therefore its own input NEURON, and the
    // T rows are the axis a spike code encodes on.
    //
    // Before this, the window's own sample axis WAS the output's row axis and T did not
    // exist. Latency coding then compared each row index against a spike time derived
    // from that row's own amplitude, so it fired only on coincidence: measured 1.07
    // spikes per 256-sample window (99.6% zeros), and 0 spikes on speech-like input.
    if (time_steps < 2)
    {
        throw std::invalid_argument(
            "encode_sample: time_steps must be >= 2 (got " + std::to_string(time_steps) +
            "); a single step leaves rate and latency coding no axis to encode on. Set "
            "model.snn_time_steps in the profile.");
    }

    const nn::Index T = static_cast<nn::Index>(time_steps);
    const nn::Index F = sample.size();

    Tensor encoded(T, F);
    encoded.set_zero();

    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();
    for (nn::Index i = 0; i < F; ++i)
    {
        min_v = std::min(min_v, sample.at(i));
        max_v = std::max(max_v, sample.at(i));
    }
    const float range = std::max(max_v - min_v, 1e-6f);

    if (encoding == "direct")
    {
        // Constant-current input: the analog value is presented at every step, which is
        // the standard way to feed a real-valued vector to an LIF stack. The membrane
        // still integrates it over T, so `direct` is not a no-op the way it was at T=1.
        for (nn::Index t = 0; t < T; ++t)
            for (nn::Index f = 0; f < F; ++f) encoded.at(t, f) = sample.at(f);
        return encoded;
    }

    if (encoding == "poisson")
    {
        // Rate coding: T independent Bernoulli draws per feature, so the spike COUNT
        // over the window carries the amplitude. Full-range min-max normalization (not
        // max-only) keeps sub-mean, post-z-score samples at a real firing probability
        // instead of clamping half the signal to 0.
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (nn::Index t = 0; t < T; ++t)
        {
            for (nn::Index f = 0; f < F; ++f)
            {
                const float p = std::clamp((sample.at(f) - min_v) / range, 0.0f, 1.0f);
                encoded.at(t, f) = (dist(rng) < p) ? 1.0f : 0.0f;
            }
        }
        return encoded;
    }

    if (encoding == "latency")
    {
        // Canonical time-to-first-spike: each feature fires exactly ONCE, at
        // t_f = round((1 - scaled_f) * (T - 1)), so larger values fire earlier. This is
        // the textbook form t_i = (I_max - I_i)/I_max * T_max, now expressible because
        // features and simulation steps live on separate axes.
        for (nn::Index f = 0; f < F; ++f)
        {
            const float scaled = std::clamp((sample.at(f) - min_v) / range, 0.0f, 1.0f);
            const auto t_spike =
                static_cast<nn::Index>(std::llround((1.0f - scaled) * static_cast<float>(T - 1)));
            encoded.at(std::clamp<nn::Index>(t_spike, 0, T - 1), f) = 1.0f;
        }
        return encoded;
    }

    throw std::invalid_argument("Unsupported encoding token: " + encoding);
}

auto make_reconstruction_target(const Tensor& sample, int time_steps) -> Tensor
{
    // The target is the ORIGINAL analog window, held constant across the simulation.
    // Reconstructing the *encoded* signal instead (what this did until 2026-09-22) makes
    // the error incomparable between encodings: a model that learns nothing scores
    // MSE 1.000 under direct, 0.247 under poisson and 0.0038 under latency, purely from
    // the target's variance. With `encoding` as a GA gene that handed the search a 261x
    // free win for picking latency.
    const nn::Index T = static_cast<nn::Index>(time_steps);
    const nn::Index F = sample.size();
    Tensor target(T, F);
    for (nn::Index t = 0; t < T; ++t)
        for (nn::Index f = 0; f < F; ++f) target.at(t, f) = sample.at(f);
    return target;
}

auto reduce_time_major_output(const Tensor& output, int time_steps) -> Tensor
{
    const nn::Index T = static_cast<nn::Index>(time_steps);
    if (T <= 0 || output.rows() % T != 0)
    {
        throw std::invalid_argument(
            "reduce_time_major_output: output rows (" + std::to_string(output.rows()) +
            ") is not a multiple of time_steps (" + std::to_string(time_steps) + ")");
    }
    const nn::Index B = output.rows() / T;
    const nn::Index F = output.cols();

    Tensor reduced(B, F);
    reduced.set_zero();
    for (nn::Index t = 0; t < T; ++t)
        for (nn::Index b = 0; b < B; ++b)
            for (nn::Index f = 0; f < F; ++f)
                reduced.at(b, f) += output.at(t * B + b, f) / static_cast<float>(T);
    return reduced;
}

auto flatten_time_series(const Tensor& sample) -> Tensor
{
    Tensor flat(1, sample.rows() * sample.cols());
    nn::Index k = 0;
    for (nn::Index t = 0; t < sample.rows(); ++t)
    {
        for (nn::Index d = 0; d < sample.cols(); ++d)
        {
            flat.at(0, k++) = sample.at(t * sample.cols() + d);
        }
    }
    return flat;
}

auto unflatten_time_series(const Tensor& flat, nn::Index rows, nn::Index cols) -> Tensor
{
    Tensor sample(rows, cols);
    nn::Index k = 0;
    for (nn::Index t = 0; t < rows; ++t)
    {
        for (nn::Index d = 0; d < cols; ++d)
        {
            sample.at(t * sample.cols() + d) = flat.at(0, k++);
        }
    }
    return sample;
}

// 3-tap {0.25, 0.5, 0.25} smoothing along the SIGNAL axis. After the 2026-09-22 layout
// change the encoded tensor is (T, F) with F = the window's consecutive samples, so the
// waveform now runs along the columns; smoothing along rows would instead blur the
// simulation steps against each other, which is not what this pre-process means.
static auto conv1d_temporal_smooth(const Tensor& sample) -> Tensor
{
    Tensor out(sample.rows(), sample.cols());
    out.set_zero();

    if (sample.rows() == 0 || sample.cols() == 0)
    {
        return out;
    }

    for (nn::Index t = 0; t < sample.rows(); ++t)
    {
        for (nn::Index d = 0; d < sample.cols(); ++d)
        {
            const nn::Index d_prev = (d == 0) ? 0 : (d - 1);
            const nn::Index d_next = (d + 1 < sample.cols()) ? (d + 1) : d;
            const float v = 0.25f * sample.at(t, d_prev) + 0.5f * sample.at(t, d) +
                            0.25f * sample.at(t, d_next);

            out.at(t, d) = v;
        }
    }

    return out;
}

// Core of the `recurrent` architecture transform. Keeps BOTH the spike train and
// the pre-reset membrane value v[t] at every step, so an inspector can plot the
// real membrane trajectory (Axis B: the 256 window samples ARE the time steps
// here — v[t] = alpha*v[t-1] + x[t] - s[t-1]*v_th). `recurrent_lif_encode`
// discards `v_mem`; `recurrent_lif_trace` (binding) returns it.
static auto recurrent_lif_run(const Tensor& sample, float alpha, float v_th) -> RecurrentLifTrace
{
    Tensor spikes(sample.rows(), sample.cols());
    spikes.set_zero();
    Tensor v_mem(sample.rows(), sample.cols());
    v_mem.set_zero();

    Tensor v_prev(1, sample.cols());
    v_prev.set_zero();
    Tensor s_prev(1, sample.cols());
    s_prev.set_zero();

    const float stable_alpha = std::clamp(alpha, 0.0f, 0.9999f);
    const float stable_vth = std::max(v_th, 1e-4f);

    for (nn::Index t = 0; t < sample.rows(); ++t)
    {
        Tensor x_t = sample.row(t);
        Tensor v_t = (v_prev * stable_alpha) + x_t - (s_prev * stable_vth);
        Tensor s_t(1, sample.cols());
        for (nn::Index d = 0; d < sample.cols(); ++d)
        {
            v_mem.at(t, d) = v_t.at(0, d);
            s_t.at(0, d) = v_t.at(0, d) >= stable_vth ? 1.0f : 0.0f;
            spikes.at(t, d) = s_t.at(0, d);
        }
        v_prev = v_t;
        s_prev = s_t;
    }

    return {std::move(spikes), std::move(v_mem)};
}

static auto recurrent_lif_encode(const Tensor& sample, float alpha, float v_th) -> Tensor
{
    return recurrent_lif_run(sample, alpha, v_th).spikes;
}

auto recurrent_lif_trace(const Tensor& encoded, float alpha, float v_th) -> RecurrentLifTrace
{
    return recurrent_lif_run(encoded, alpha, v_th);
}

auto apply_snn_architecture_transform(
    const Tensor& encoded, const std::string& architecture, float alpha, float v_th) -> Tensor
{
    if (architecture == "conv1d") return conv1d_temporal_smooth(encoded);
    if (architecture == "recurrent") return recurrent_lif_encode(encoded, alpha, v_th);
    return encoded;
}

auto to_lstm_frames(const Tensor& sample, int frame_size) -> Tensor
{
    if (frame_size <= 0)
    {
        throw std::invalid_argument("to_lstm_frames: frame_size must be positive");
    }

    const auto total = sample.size();
    const auto frame = static_cast<nn::Index>(frame_size);
    if (total % frame != 0)
    {
        throw std::invalid_argument("to_lstm_frames: sample length (" + std::to_string(total) +
                                    ") is not divisible by frame_size (" +
                                    std::to_string(frame_size) + ")");
    }

    if (frame == 1) return sample; // already one scalar per timestep

    const nn::Index steps = total / frame;

    // Column-major storage: reshaping to (frame, steps) makes element (d, t)
    // land on flat index d + t*frame, i.e. sample[t*frame + d] — the consecutive
    // framing we want, laid out D-major. Transposing gives (steps, frame).
    Tensor d_major = sample;
    d_major.reshape({frame, steps});
    return d_major.transpose();
}

} // namespace meeting01

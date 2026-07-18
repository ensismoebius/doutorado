#include "../include/E04Encoding.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

namespace e04
{

auto encode_sample(const Tensor& sample, const std::string& encoding, std::uint32_t seed) -> Tensor
{
    Tensor encoded(sample.rows(), sample.cols());
    encoded.set_zero();

    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();
    for (nn::Index i = 0; i < sample.size(); ++i)
    {
        min_v = std::min(min_v, sample.at(i));
        max_v = std::max(max_v, sample.at(i));
    }
    const float range = std::max(max_v - min_v, 1e-6f);

    if (encoding == "direct")
    {
        return sample;
    }

    if (encoding == "poisson")
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        const float max_only = std::max(max_v, 1e-6f);
        for (nn::Index t = 0; t < sample.rows(); ++t)
        {
            for (nn::Index d = 0; d < sample.cols(); ++d)
            {
                const float p = std::clamp(sample.at(t * sample.cols() + d) / max_only, 0.0f, 1.0f);
                encoded.at(t, d) = (dist(rng) < p) ? 1.0f : 0.0f;
            }
        }
        return encoded;
    }

    if (encoding == "latency")
    {
        const nn::Index T = sample.rows();
        for (nn::Index t = 0; t < sample.rows(); ++t)
        {
            for (nn::Index d = 0; d < sample.cols(); ++d)
            {
                const float scaled = (sample.at(t * sample.cols() + d) - min_v) / range;
                const nn::Index t_spike = static_cast<nn::Index>(std::llround(
                    (1.0f - scaled) * static_cast<float>(std::max<nn::Index>(1, T - 1))));
                encoded.at(t, d) = (t >= t_spike) ? 1.0f : 0.0f;
            }
        }
        return encoded;
    }

    throw std::invalid_argument("Unsupported encoding token: " + encoding);
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
            const nn::Index t_prev = (t == 0) ? 0 : (t - 1);
            const nn::Index t_next = (t + 1 < sample.rows()) ? (t + 1) : t;
            const float v = 0.25f * sample.at(t_prev, d) + 0.5f * sample.at(t, d) +
                            0.25f * sample.at(t_next, d);

            out.at(t, d) = v;
        }
    }

    return out;
}

static auto recurrent_lif_encode(const Tensor& sample, float alpha, float v_th) -> Tensor
{
    Tensor spikes(sample.rows(), sample.cols());
    spikes.set_zero();

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
            s_t.at(0, d) = v_t.at(0, d) >= stable_vth ? 1.0f : 0.0f;
            spikes.at(t, d) = s_t.at(0, d);
        }
        v_prev = v_t;
        s_prev = s_t;
    }

    return spikes;
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

} // namespace e04

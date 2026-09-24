// bind_meeting01.cpp — nn_microscope.meeting01
//
// Live recomputation of the meeting01 window pipeline for the inspection GUI
// (FIXME §1, §9, §43-B). Every function here calls the exact meeting01 C++
// implementation — encodings, architecture transforms, the SNN autoencoder —
// never a reimplementation (FIXME §3, §198).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/Meeting01Config.hpp"
#include "../include/Meeting01Dataset.hpp"
#include "../include/Meeting01Encoding.hpp"
#include "../include/Meeting01Training.hpp"
#include "layers/Layers.hpp"
#include "models/autoencoder/ProtocolSpikingAutoencoder.hpp"
#include "nlohmann/json.hpp"
#include "serialization/NetworkSerializer.hpp"
#include "tensor_bridge.hpp"

namespace py = pybind11;
using nn_microscope::from_numpy;
using nn_microscope::to_numpy;

namespace
{

meeting01::Meeting01Config load_config(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::invalid_argument("meeting01: cannot open config '" + path +
                                    "' (remedy: pass the path to meeting01-loso.json).");
    }
    nlohmann::json j;
    in >> j;
    // The LOSO profile is the nested form; fall back to flat for older profiles.
    if (j.contains("experiment") && j.contains("dataset") && j.contains("training"))
    {
        return meeting01::Meeting01Config::from_nested_json(j);
    }
    return meeting01::Meeting01Config::from_flat_json(j);
}

py::dict window_meta_to_dict(const meeting01::WindowMetadata& m)
{
    py::dict d;
    d["speaker"] = m.speaker;
    d["speaker_id"] = m.speaker_id;
    d["recording_id"] = m.recording_id;
    d["window_id"] = m.window_id;
    d["source_window_index"] = m.source_window_index;
    d["digit"] = m.digit;
    d["valid_length"] = m.valid_length;
    return d;
}

py::list samples_to_list(const std::vector<nn::Tensor>& samples)
{
    py::list out;
    for (const auto& s : samples)
    {
        out.append(to_numpy(s));
    }
    return out;
}

py::list metas_to_list(const std::vector<meeting01::WindowMetadata>& metas)
{
    py::list out;
    for (const auto& m : metas)
    {
        out.append(window_meta_to_dict(m));
    }
    return out;
}

} // namespace

void bind_meeting01(py::module_& parent)
{
    py::module_ m = parent.def_submodule(
        "meeting01", "Live meeting01 window pipeline (encodings, transforms, SNN-AE forward).");

    // -- encodings + transforms (Meeting01Encoding.hpp) -----------------------
    // `time_steps` is deliberately required and deliberately ahead of `seed`: the encoded
    // tensor's shape is (time_steps, window_size), so a caller that omits it would get a
    // silently different tensor than the experiment produced. Breaking such a caller
    // loudly is the point.
    m.def(
        "encode_sample",
        [](const py::array& sample, const std::string& encoding, int time_steps, std::uint32_t seed)
        {
            return to_numpy(
                meeting01::encode_sample(from_numpy(sample), encoding, seed, time_steps));
        },
        py::arg("sample"),
        py::arg("encoding"),
        py::arg("time_steps"),
        py::arg("seed") = 0,
        "direct | poisson | latency, exactly as the experiment encodes it. Returns a "
        "(time_steps, window_size) time-major tensor; time_steps must be >= 2.");

    m.def(
        "make_reconstruction_target",
        [](const py::array& sample, int time_steps)
        { return to_numpy(meeting01::make_reconstruction_target(from_numpy(sample), time_steps)); },
        py::arg("sample"),
        py::arg("time_steps"),
        "The training target every model in the experiment is scored against: the ORIGINAL "
        "z-scored window repeated across the steps, never the encoded one.");

    m.def(
        "make_activity_mask",
        [](int valid_length, int window_size)
        { return to_numpy(meeting01::make_activity_mask(valid_length, window_size)); },
        py::arg("valid_length"),
        py::arg("window_size"),
        "1.0 for the first valid_length rows (real signal), 0.0 for the zero-padded tail "
        "(FSDD/AudioMNIST's trailing partial window only -- see build_split's "
        "*_meta[i]['valid_length']). Pass through make_reconstruction_target/to_lstm_frames "
        "exactly like the window itself to get a mask of matching shape for a masked loss.");

    m.def(
        "reduce_time_major_output",
        [](const py::array& output, int time_steps)
        { return to_numpy(meeting01::reduce_time_major_output(from_numpy(output), time_steps)); },
        py::arg("output"),
        py::arg("time_steps"),
        "Temporal-averaging readout: collapses a (T, F) model output to (1, F).");

    m.def(
        "apply_snn_architecture_transform",
        [](const py::array& encoded, const std::string& architecture, float alpha, float v_th)
        {
            return to_numpy(meeting01::apply_snn_architecture_transform(
                from_numpy(encoded), architecture, alpha, v_th));
        },
        py::arg("encoded"),
        py::arg("architecture"),
        py::arg("alpha") = 0.9F,
        py::arg("v_th") = 1.0F,
        "dense (pass-through) | conv1d (3-tap) | recurrent (LIF sweep).");

    m.def(
        "recurrent_lif_trace",
        [](const py::array& encoded, float alpha, float v_th)
        {
            const auto tr = meeting01::recurrent_lif_trace(from_numpy(encoded), alpha, v_th);
            py::dict d;
            d["spikes"] = to_numpy(tr.spikes);
            d["v_mem"] = to_numpy(tr.v_mem);
            return d;
        },
        py::arg("encoded"),
        py::arg("alpha") = 0.9F,
        py::arg("v_th") = 1.0F,
        "recurrent-transform spike train PLUS the per-step membrane trajectory "
        "v[t] = alpha*v[t-1] + x[t] - s[t-1]*v_th (the window samples are the time steps).");

    m.def(
        "flatten_time_series",
        [](const py::array& sample)
        { return to_numpy(meeting01::flatten_time_series(from_numpy(sample))); },
        py::arg("sample"));

    m.def(
        "to_lstm_frames",
        [](const py::array& sample, int frame_size)
        { return to_numpy(meeting01::to_lstm_frames(from_numpy(sample), frame_size)); },
        py::arg("sample"),
        py::arg("frame_size"));

    // -- windowed LOSO split (Meeting01Dataset.hpp) -------------------------
    m.def(
        "build_split",
        [](const std::string& config_path, const std::string& dataset, int cv_fold)
        {
            const auto cfg = load_config(config_path);
            const auto split = meeting01::build_split(cfg, dataset, cv_fold);
            py::dict d;
            d["train_samples"] = samples_to_list(split.train_samples);
            d["val_samples"] = samples_to_list(split.val_samples);
            d["test_samples"] = samples_to_list(split.test_samples);
            d["train_meta"] = metas_to_list(split.train_meta);
            d["val_meta"] = metas_to_list(split.val_meta);
            d["test_meta"] = metas_to_list(split.test_meta);
            d["val_labels"] = split.val_labels;
            d["test_labels"] = split.test_labels;
            d["train_speakers"] = split.train_speakers;
            d["val_speaker"] = split.val_speaker;
            d["test_speaker"] = split.test_speaker;
            return d;
        },
        py::arg("config_path"),
        py::arg("dataset"),
        py::arg("cv_fold") = -1,
        "Speaker-disjoint nested-LOSO fold. Each *_samples entry is a (window_size, 1) array, "
        "z-scored per window exactly as the experiment loads it.");

    // -- SNN autoencoder forward (reload trained weights, no retrain) --------
    m.def(
        "snn_ae_forward",
        [](const std::string& config_path,
            float alpha,
            float v_th,
            const std::string& architecture,
            const std::string& encoder_npz,
            const std::string& decoder_npz,
            const py::array& flat_window,
            const std::string& encoding,
            std::uint32_t seed)
        {
            const auto cfg = load_config(config_path);
            auto ae_cfg = meeting01::make_snn_cfg(cfg, alpha, v_th);
            nn::models::autoencoder::ProtocolSpikingAutoencoder model(ae_cfg);
            // Load trained weights INTO the AE's own (correct) topology — do not
            // rebuild layers from the checkpoint's architecture metadata, which
            // for a LifBPTT encoder saved by an older build omits the LIF lines.
            if (!NetworkSerializer::loadParametersInto(model.encoder_, encoder_npz))
            {
                throw std::runtime_error("snn_ae_forward: failed to load encoder '" + encoder_npz +
                                         "' (remedy: run the LOSO pipeline with save_models=true "
                                         "to emit *_encoder.npz).");
            }
            if (!NetworkSerializer::loadParametersInto(model.decoder_, decoder_npz))
            {
                throw std::runtime_error(
                    "snn_ae_forward: failed to load decoder '" + decoder_npz + "'.");
            }

            // Reproduce the experiment's per-window recompute chain:
            //   encode_sample -> apply_snn_architecture_transform -> network
            // The encoded tensor is fed in AS IS, time-major (T*B, F) with B = 1, because
            // that is what the training path now does; flattening it to one row was the
            // T=1 layout and would reproduce a model state the experiment never trains.
            const int steps = cfg.model.time_steps;
            nn::Tensor sample = from_numpy(flat_window);
            nn::Tensor enc = meeting01::encode_sample(sample, encoding, seed, steps);
            enc = meeting01::apply_snn_architecture_transform(enc, architecture, alpha, v_th);

            model.reset_state();
            nn::Tensor latent = model.encode(enc, /*requires_grad=*/false);
            nn::Tensor recon = model.decode(latent, /*requires_grad=*/false);
            // The window-level reconstruction the metrics use is the temporal average.
            nn::Tensor recon_window = meeting01::reduce_time_major_output(recon, steps);

            // Per-layer encoder trace for the SNN Lab / SNN-3D views (FIXME §15, §18).
            // Sequential caches every layer output; each LifBPTT keeps its post-forward
            // membrane snapshot — with time_steps = 16 that snapshot is the membrane at
            // the LAST step, not a trajectory; `output` holds all T steps.
            auto encoder_trace = [](const nn::Sequential& seq)
            {
                py::list layers;
                for (size_t i = 0; i < seq.layers.size(); ++i)
                {
                    py::dict ld;
                    auto* raw = seq.layers[i].get();
                    if (auto* lin = dynamic_cast<nn::Linear*>(raw))
                    {
                        ld["type"] = "linear";
                        ld["weight"] = to_numpy(lin->weight);
                    }
                    else if (auto* lif = dynamic_cast<nn::LifBPTT*>(raw))
                    {
                        ld["type"] = "lif";
                        ld["v_mem"] = to_numpy(lif->v_mem);
                        ld["voltage_threshold"] =
                            static_cast<double>(lif->voltage_threshold.at(0, 0));
                    }
                    else
                    {
                        ld["type"] = "other";
                    }
                    if (i < seq.outputs.size()) ld["output"] = to_numpy(seq.outputs[i]);
                    layers.append(ld);
                }
                return layers;
            };

            py::dict out;
            out["latent"] = to_numpy(latent);
            out["reconstruction"] = to_numpy(recon);
            out["reconstruction_window"] = to_numpy(recon_window);
            out["target_window"] = to_numpy(meeting01::make_reconstruction_target(sample, steps));
            out["encoded_input"] = to_numpy(enc);
            out["time_steps"] = steps;
            // Both halves are traced with the same helper so the GUI can draw the
            // encoder and decoder as one continuous graph (FIXME §18, §19).
            out["encoder_layers"] = encoder_trace(model.encoder_);
            out["decoder_layers"] = encoder_trace(model.decoder_);
            return out;
        },
        py::arg("config_path"),
        py::arg("alpha"),
        py::arg("v_th"),
        py::arg("architecture"),
        py::arg("encoder_npz"),
        py::arg("decoder_npz"),
        py::arg("flat_window"),
        py::arg("encoding") = "direct",
        py::arg("seed") = 0,
        "Latent + reconstruction for one window, from the retrained-winner .npz. "
        "`reconstruction` is the raw (T, F) output, `reconstruction_window` its temporal "
        "average against `target_window` (the original z-scored window). Plus "
        "encoder_layers and decoder_layers: per-layer {type, output, weight|v_mem, "
        "voltage_threshold}; `output` spans all T steps, `v_mem` is the membrane at the "
        "last step only.");
}

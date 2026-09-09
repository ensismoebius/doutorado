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
    m.def(
        "encode_sample",
        [](const py::array& sample, const std::string& encoding, std::uint32_t seed)
        { return to_numpy(meeting01::encode_sample(from_numpy(sample), encoding, seed)); },
        py::arg("sample"),
        py::arg("encoding"),
        py::arg("seed") = 0,
        "direct | poisson | latency, exactly as the experiment encodes it.");

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
            if (!NetworkSerializer::loadNetwork(model.encoder_, encoder_npz))
            {
                throw std::runtime_error("snn_ae_forward: failed to load encoder '" + encoder_npz +
                                         "' (remedy: run the LOSO pipeline with save_models=true "
                                         "to emit *_encoder.npz).");
            }
            if (!NetworkSerializer::loadNetwork(model.decoder_, decoder_npz))
            {
                throw std::runtime_error(
                    "snn_ae_forward: failed to load decoder '" + decoder_npz + "'.");
            }

            // Reproduce the experiment's per-window recompute chain:
            //   encode_sample -> apply_snn_architecture_transform -> flatten_time_series
            nn::Tensor sample = from_numpy(flat_window);
            nn::Tensor enc = meeting01::encode_sample(sample, encoding, seed);
            enc = meeting01::apply_snn_architecture_transform(enc, architecture, alpha, v_th);
            nn::Tensor flat = meeting01::flatten_time_series(enc);

            nn::Tensor latent = model.encode(flat, /*requires_grad=*/false);
            nn::Tensor recon = model.decode(latent, /*requires_grad=*/false);

            py::dict out;
            out["latent"] = to_numpy(latent);
            out["reconstruction"] = to_numpy(recon);
            out["encoded_input"] = to_numpy(flat);
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
        "Spike / membrane traces are added in a later milestone.");
}

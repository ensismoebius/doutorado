// bind_thesis.cpp — nn_microscope.thesis
//
// Live recomputation of the thesis feature + paraconsistent pipeline
// (FIXME §2, §12, §14, §55). Handcrafted feature extraction and the
// paraconsistent EPC/alpha/beta score both call the exact thesis
// implementation.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ThesisConfig.hpp"
#include "ThesisDataset.hpp"
#include "ThesisFeatureExtraction.hpp"
#include "ThesisParaconsistent.hpp"
#include "ThesisSample.hpp"
#include "tensor_bridge.hpp"

namespace py = pybind11;
using nn_microscope::to_numpy;

namespace
{

thesis::ThesisConfig::HandcraftedConfig make_handcrafted_cfg(const std::string& transform,
    const std::string& scale,
    const std::vector<std::string>& descriptors,
    int dtwpt_level,
    const std::string& wavelet,
    bool cepstral)
{
    thesis::ThesisConfig::HandcraftedConfig cfg;
    cfg.transform = transform;
    cfg.scale = scale;
    cfg.descriptors = descriptors;
    cfg.dtwpt_level = dtwpt_level;
    cfg.wavelet = wavelet;
    cfg.cepstral = cepstral;
    return cfg;
}

} // namespace

void bind_thesis(py::module_& parent)
{
    py::module_ m = parent.def_submodule(
        "thesis", "Live thesis handcrafted-feature + paraconsistent pipeline (FIXME sec 2/12/14).");

    py::class_<thesis::ParaconsistentScore>(m, "ParaconsistentScore")
        .def_readonly("label", &thesis::ParaconsistentScore::label)
        .def_readonly("alpha", &thesis::ParaconsistentScore::alpha)
        .def_readonly("beta", &thesis::ParaconsistentScore::beta)
        .def_readonly("g1", &thesis::ParaconsistentScore::g1)
        .def_readonly("g2", &thesis::ParaconsistentScore::g2)
        .def_readonly("d_truth", &thesis::ParaconsistentScore::d_truth)
        .def_readonly("d_penalized", &thesis::ParaconsistentScore::d_penalized)
        .def("__repr__",
            [](const thesis::ParaconsistentScore& s)
            {
                return "<ParaconsistentScore '" + s.label +
                       "' d_penalized=" + std::to_string(s.d_penalized) + ">";
            });

    m.def(
        "extract_handcrafted",
        [](const std::vector<double>& signal,
            double sample_rate,
            const std::string& transform,
            const std::string& scale,
            const std::vector<std::string>& descriptors,
            int dtwpt_level,
            const std::string& wavelet,
            bool cepstral)
        {
            const auto cfg =
                make_handcrafted_cfg(transform, scale, descriptors, dtwpt_level, wavelet, cepstral);
            return thesis::extract_handcrafted(signal, cfg, sample_rate);
        },
        py::arg("signal"),
        py::arg("sample_rate"),
        py::arg("transform") = "dtwpt",
        py::arg("scale") = "lfcc",
        py::arg("descriptors") = std::vector<std::string>{"energy", "zcr", "entropy", "teager"},
        py::arg("dtwpt_level") = 4,
        py::arg("wavelet") = "daub4",
        py::arg("cepstral") = false,
        "DTWPT (or cepstral) handcrafted feature vector for one signal — "
        "thesis::extract_handcrafted.");

    m.def(
        "paraconsistent_score",
        [](const std::vector<std::vector<double>>& feature_vectors,
            const std::vector<int>& subject_ids,
            const std::string& label)
        {
            if (feature_vectors.size() != subject_ids.size())
            {
                throw std::invalid_argument(
                    "paraconsistent_score: feature_vectors and subject_ids must have the same "
                    "length (remedy: pass one subject id per feature vector).");
            }
            std::vector<thesis::ThesisSample> samples(feature_vectors.size());
            for (std::size_t i = 0; i < subject_ids.size(); ++i)
            {
                samples[i].subject_id = subject_ids[i];
            }
            thesis::FeatureSet fs;
            fs.label = label;
            fs.vectors = feature_vectors;
            return thesis::score_feature_set(samples, fs);
        },
        py::arg("feature_vectors"),
        py::arg("subject_ids"),
        py::arg("label") = "adhoc",
        "alpha/beta/G1/G2/D_truth/D_penalized for a feature set — thesis::score_feature_set. "
        "Grouping is by subject_id; d_penalized = d_truth + (2 - sqrt(2)) * |g2|.");

    m.attr("K_CONTRADICTION_PENALTY") = thesis::kContradictionPenalty;

    // -- dataset -> features -> paraconsistent ranking (the full Phase 00 chain) -------
    using ViewPtr = std::shared_ptr<thesis::ThesisDatasetView>;

    py::class_<thesis::ThesisDatasetView, ViewPtr>(m, "DatasetView")
        .def_property_readonly(
            "n_samples", [](const thesis::ThesisDatasetView& v) { return v.samples.size(); })
        .def_readonly("n_subjects", &thesis::ThesisDatasetView::n_subjects)
        .def_readonly("n_stimuli", &thesis::ThesisDatasetView::n_stimuli)
        .def_property_readonly("subject_ids",
            [](const thesis::ThesisDatasetView& v)
            {
                std::vector<int> ids;
                ids.reserve(v.samples.size());
                for (const auto& s : v.samples) ids.push_back(s.subject_id);
                return ids;
            })
        .def_property_readonly("stimuli",
            [](const thesis::ThesisDatasetView& v)
            {
                std::vector<int> st;
                st.reserve(v.samples.size());
                for (const auto& s : v.samples) st.push_back(s.stimulus);
                return st;
            })
        .def(
            "sample",
            [](const thesis::ThesisDatasetView& v, std::size_t i)
            {
                if (i >= v.samples.size())
                {
                    throw std::out_of_range("DatasetView.sample: index " + std::to_string(i) +
                                            " >= n_samples " + std::to_string(v.samples.size()));
                }
                const auto& s = v.samples[i];
                py::dict d;
                d["subject_id"] = s.subject_id;
                d["stimulus"] = s.stimulus;
                d["text_phrase"] = s.text_phrase;
                d["audio"] = to_numpy(s.audio); // (N_audio, 1)
                d["eeg"] = to_numpy(s.eeg);     // (N_channels, N_samples)
                return d;
            },
            py::arg("index"),
            "Raw audio + EEG tensors for one sample, as ndarrays.");

    py::class_<thesis::FeatureSet>(m, "FeatureSet")
        .def_readonly("label", &thesis::FeatureSet::label)
        .def_readonly("vectors", &thesis::FeatureSet::vectors)
        .def("__repr__",
            [](const thesis::FeatureSet& f)
            { return "<FeatureSet '" + f.label + "' x" + std::to_string(f.vectors.size()) + ">"; });

    m.def(
        "load_dataset",
        [](const std::string& root, const std::string& modality, int max_samples)
        {
            thesis::ThesisConfig::Dataset d;
            d.root = root;
            d.modality = modality;
            d.max_samples = max_samples;
            return std::make_shared<thesis::ThesisDatasetView>(thesis::load_dataset(d));
        },
        py::arg("root"),
        py::arg("modality") = "eeg",
        py::arg("max_samples") = 0,
        "thesis::load_dataset — root may be a subject directory or a .sqlite file "
        "('~' is expanded). modality: voice | eeg | fused.");

    m.def(
        "extract_handcrafted_features",
        [](const ViewPtr& view,
            const std::string& modality,
            const std::string& transform,
            const std::string& scale,
            const std::vector<std::string>& descriptors,
            int dtwpt_level,
            const std::string& wavelet,
            bool cepstral,
            const std::string& fusion_mode,
            std::uint32_t seed)
        {
            thesis::ThesisConfig::FeatureExtraction fx;
            fx.strategy = "handcrafted";
            fx.handcrafted =
                make_handcrafted_cfg(transform, scale, descriptors, dtwpt_level, wavelet, cepstral);
            const thesis::ThesisConfig::Training training{};
            return thesis::extract_features(*view, fx, training, modality, fusion_mode, seed);
        },
        py::arg("view"),
        py::arg("modality") = "eeg",
        py::arg("transform") = "dtwpt",
        py::arg("scale") = "lfcc",
        py::arg("descriptors") = std::vector<std::string>{"energy", "zcr", "entropy", "teager"},
        py::arg("dtwpt_level") = 4,
        py::arg("wavelet") = "daub4",
        py::arg("cepstral") = false,
        py::arg("fusion_mode") = "late",
        py::arg("seed") = 42U,
        "thesis::extract_features with the handcrafted strategy — one FeatureSet per "
        "modality/config, vectors aligned with view.samples.");

    m.def(
        "rank_feature_sets",
        [](const ViewPtr& view, const std::vector<thesis::FeatureSet>& sets)
        { return thesis::rank_feature_sets(view->samples, sets); },
        py::arg("view"),
        py::arg("feature_sets"),
        "thesis::rank_feature_sets — ParaconsistentScore per set, ascending by d_penalized.");
}

// bind_thesis.cpp — nn_microscope.thesis
//
// Live recomputation of the thesis feature + paraconsistent pipeline
// (FIXME §2, §12, §14, §55). Handcrafted feature extraction and the
// paraconsistent EPC/alpha/beta score both call the exact thesis
// implementation.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "ThesisConfig.hpp"
#include "ThesisFeatureExtraction.hpp"
#include "ThesisParaconsistent.hpp"
#include "ThesisSample.hpp"

namespace py = pybind11;

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
}

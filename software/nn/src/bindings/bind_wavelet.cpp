// bind_wavelet.cpp — nn_microscope.wavelet
//
// Exposes the repository's actual Mallat-algorithm wavelet transform
// (wavelets::malat) and subband-energy extraction. The GUI's Wavelet Lab
// (FIXME §10, §11) recomputes decompositions through this — never a
// PyWavelets substitute (FIXME §3, §198).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "wavelet/WaveletTraits.hpp"
#include "wavelet/WaveletTransformResults.hpp"
#include "wavelet/waveletOperations.hpp"

namespace py = pybind11;

namespace
{

// Name -> low-pass decomposition filter. Matches the string table thesis uses
// in ThesisHandcraftedFeatures.cpp::wavelet_filter. "daubN" -> N-tap Daubechies
// (pywt db(N/2)); "haar" == db1.
std::span<const double> wavelet_filter(const std::string& name)
{
    using namespace wavelets;
    if (name == "haar" || name == "db1") return WaveletTraits<Haar>::coeffs;
#define NNM_DAUB(N) \
    if (name == "daub" #N) return WaveletTraits<Daub##N>::coeffs
    NNM_DAUB(4);
    NNM_DAUB(6);
    NNM_DAUB(8);
    NNM_DAUB(10);
    NNM_DAUB(12);
    NNM_DAUB(14);
    NNM_DAUB(16);
    NNM_DAUB(18);
    NNM_DAUB(20);
    NNM_DAUB(22);
    NNM_DAUB(24);
    NNM_DAUB(26);
    NNM_DAUB(28);
    NNM_DAUB(30);
    NNM_DAUB(32);
    NNM_DAUB(34);
    NNM_DAUB(36);
    NNM_DAUB(38);
    NNM_DAUB(40);
    NNM_DAUB(42);
    NNM_DAUB(44);
    NNM_DAUB(46);
#undef NNM_DAUB
    throw std::invalid_argument(
        "unknown wavelet '" + name +
        "'. Known: haar, daub4, daub6, ..., daub46 "
        "(remedy: pass one of these names, matching ThesisHandcraftedFeatures.cpp).");
}

wavelets::TransformMode parse_mode(const std::string& mode)
{
    if (mode == "packet") return wavelets::PACKET_WAVELET;
    if (mode == "regular") return wavelets::REGULAR_WAVELET;
    throw std::invalid_argument("mode must be 'packet' or 'regular', got '" + mode + "'");
}

} // namespace

void bind_wavelet(py::module_& parent)
{
    py::module_ m = parent.def_submodule(
        "wavelet", "Repository wavelet transform (wavelets::malat) — FIXME sec 10/11.");

    py::class_<wavelets::WaveletTransformResults>(m, "WaveletResult")
        .def_readonly("packet", &wavelets::WaveletTransformResults::packet)
        .def_readonly("levels", &wavelets::WaveletTransformResults::levelsOfTransformation)
        .def_property_readonly("transformed_signal",
            [](wavelets::WaveletTransformResults& r) { return r.transformedSignal; })
        .def(
            "wavelet_transforms",
            [](wavelets::WaveletTransformResults& r, int detail_index)
            { return r.get_wavelet_transforms(detail_index); },
            py::arg("detail_index") = -1,
            "detail_index: -1 whole signal, 0 approximation, >=1 the k-th detail band")
        .def(
            "packet_transforms",
            [](wavelets::WaveletTransformResults& r, long start, long end, long max_freq)
            { return r.get_wavelet_packet_transforms(start, end, max_freq); },
            py::arg("start"),
            py::arg("end"),
            py::arg("max_freq"))
        .def("packet_leaf_count",
            [](const wavelets::WaveletTransformResults& r)
            { return r.get_wavelet_packet_amount_of_parts(); });

    m.def(
        "decompose",
        [](const std::vector<double>& signal,
            const std::string& wavelet,
            const std::string& mode,
            unsigned int level)
        { return wavelets::malat(signal, wavelet_filter(wavelet), parse_mode(mode), level); },
        py::arg("signal"),
        py::arg("wavelet") = "haar",
        py::arg("mode") = "packet",
        py::arg("level") = 4,
        "Mallat-algorithm DWT / wavelet-packet transform (wavelets::malat).");

    m.def(
        "subband_energies",
        [](const wavelets::WaveletTransformResults& r, int level)
        { return wavelets::extract_subband_energies(r, level); },
        py::arg("result"),
        py::arg("level"),
        "RMS energy per subband (wavelets::extract_subband_energies).");

    m.def(
        "next_power_of_two",
        [](double n) { return wavelets::get_next_power_of_two(n); },
        py::arg("n"));
}

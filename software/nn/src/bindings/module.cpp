// module.cpp — nn_microscope: the pybind11 entry point for
// software/experiment_microscope/.
//
// FIXME sec 3 / 198: the inspection GUI must never contain a second
// implementation of the scientific pipeline. Every interactive recomputation
// links the actual meeting01 / thesis C++ code through this module.
//
// Submodules (built incrementally):
//   nn_microscope.wavelet        wavelets::malat + subband energies       [live]
//   nn_microscope.meeting01      dataset / encoding / SNN-AE forward      [planned]
//   nn_microscope.thesis         dataset / handcrafted / paraconsistent   [planned]

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_wavelet(py::module_&);
void bind_meeting01(py::module_&);
void bind_thesis(py::module_&);

PYBIND11_MODULE(nn_microscope, m)
{
    m.doc() =
        "Live recomputation bridge for the Experiment Microscope GUI. "
        "Links the meeting01 / thesis C++ libraries — no NumPy reimplementation.";
    m.attr("__version__") = "0.1.0";

    // Paraconsistent contradiction penalty, kept in lock-step with
    // ThesisParaconsistent.hpp (inline constexpr kContradictionPenalty).
    m.attr("K_CONTRADICTION_PENALTY") = 0.5857864376269049; // 2 - sqrt(2)

    bind_wavelet(m);
    bind_meeting01(m);
    bind_thesis(m);
}

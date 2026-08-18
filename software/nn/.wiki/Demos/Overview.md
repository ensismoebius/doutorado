# Demos Overview

The `src/demos/` directory contains self-contained runnable examples covering signal processing, machine learning, and spiking neural network functionality. Each demo has its own `README.md` (the authoritative source), CMakeLists target, and optional test suite.

## C++ Demos

| Demo | Description | Wiki Page |
|------|-------------|-----------|
| `fft_demo` | FFT of a two-component sine signal via FFTW3; plots time-domain and dB spectrum | [fft-demo](./fft-demo.md) |
| `wavelet_demo` | DWT and DWPT of a synthetic signal with Daub8 wavelet; saves 9 PNG plots | [wavelet-demo](./wavelet-demo.md) |
| `lfcc_feature_demo` | Batch LFCC extraction pipeline over the BaseDeDatosHablaImaginada corpus | [lfcc-feature-demo](./lfcc-feature-demo.md) |
| `resnet_classifier_demo` | Residual MLP trained on MAT audio features with CrossEntropyLoss + Adam | [resnet-classifier-demo](./resnet-classifier-demo.md) |
| `snn_speaker_demo` | End-to-end SNN speaker identification: LFCC features → Poisson encoding → residual LifBPTT | [snn-speaker-demo](./snn-speaker-demo.md) |
| `snn_spike_plotter` | Real-time ImGui/ImPlot visualisation of a two-neuron LIF chain driven by Poisson input | [snn-spike-plotter](./snn-spike-plotter.md) |
| `wpt_voice_biometrics` | WPT subband energies → Poisson encoding → residual SNN; CLI with WAV input | [wpt-voice-biometrics](./wpt-voice-biometrics.md) |
| `autoencoder_leakyrelu` | Spiking autoencoder (LifBPTT) trained end-to-end on synthetic data; validates BPTT stack | [autoencoder-leakyrelu](./autoencoder-leakyrelu.md) |

## Python Demos

| Demo | Description | Wiki Page |
|------|-------------|-----------|
| `multimodal_eeg_audio` | Multimodal EEG + audio fusion with dense/spiking autoencoders and paraconsistent analysis | [multimodal-eeg-audio](./multimodal-eeg-audio.md) |
| `snn_hyperparam_search` | Four-stage hyperparameter search for a PyTorch/snnTorch SNN autoencoder | [snn-hyperparam-search](./snn-hyperparam-search.md) |
| `voice_biometrics_snn_py` | Python WPT → spike → SNN voice biometrics with full enrol/train/identify/verify CLI | [voice-biometrics-snn-py](./voice-biometrics-snn-py.md) |

## Lecture companion (separate application)

Not under `src/demos/` and not part of the `nn` library — a standalone PySide6 application in
`software/efficient_nn_lab/`, written for a one-hour undergraduate lecture and driven live from
the LaTeX deck's PDF links.

| Application | Description | Wiki Page |
|------|-------------|-----------|
| `efficient_nn_lab` | 13 interactive demos animating BitNet ternary quantization, the STE, and SNN spiking/surrogate mechanics, one step at a time | [efficient-nn-lab](./efficient-nn-lab.md) |

## Building All C++ Demos

```bash
cd /home/ensismoebius/Repos/doutorado/software/nn
cmake --preset=max-performance
# Build each demo individually (replace <target> with the target name from each demo page)
cmake --build out/build/max-performance --target <target> -j$(nproc)
```

## Related Wiki Sections

- [Core/Layers](../Core/Layers.md) — layer implementations used by the demos
- [Concepts/Membrane-Dynamics](../Concepts/Membrane-Dynamics.md) — LIF neuron theory
- [Concepts/Time-Major-Layout](../Concepts/Time-Major-Layout.md) — SNN tensor shape convention
- [Concepts/SNN-and-Surrogate-Gradients](../Concepts/SNN-and-Surrogate-Gradients.md) — BPTT for spiking networks

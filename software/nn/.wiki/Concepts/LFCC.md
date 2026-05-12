# LFCC — Linear Frequency Cepstral Coefficients

LFCC uses a linearly-spaced filterbank (as opposed to the perceptually-motivated non-linear Mel or Bark scales) to extract spectral envelope features from speech signals. It is preferred over MFCC for **speaker verification** tasks because it preserves high-frequency components that carry speaker-discriminative glottal and vocal tract characteristics.

## Why LFCC Instead of MFCC/BARK for Speaker Verification

The Mel and Bark scales compress frequencies above ~4 kHz to model human auditory perception. This compression discards high-frequency spectral content that is irrelevant for speech *recognition* but carries **speaker-identifying** characteristics [1, 2]:

- Glottal source characteristics (jitter, shimmer)
- Higher formants (F3, F4) related to larynx and pharynx shape
- Spectral fine structure of the vocal tract above 4 kHz

LFCC maintains uniform resolution across the full frequency range, preserving these components.

| Feature | LFCC | MFCC | BARK |
|---|---|---|---|
| Filterbank spacing | Linear | Mel (logarithmic above ~1 kHz) | Bark (psychoacoustic) |
| High-freq resolution | Uniform | Compressed | Compressed |
| Perceptual motivation | None | Human hearing | Human hearing |
| Good for speech recog | Moderate | Excellent | Good |
| Good for speaker verif | **Excellent** | Good | Good |
| Computational cost | Low | Low | Low |

## Theory

### Filterbank

A bank of $K$ triangular filters with centers **equispaced** between $f_{\min}$ and $f_{\max}$:

$$f_k = f_{\min} + k \cdot \frac{f_{\max} - f_{\min}}{K + 1}, \qquad k = 1, \ldots, K$$

Each filter computes the energy of the signal in its band:

$$E_k = \sum_{\omega \in \text{band}_k} |X(\omega)|^2$$

where $X(\omega)$ is the DFT of the windowed signal frame.

### DCT Decorrelation

The log-energies are decorrelated using the same DCT step as in MFCC [3]:

$$c_n = \sum_{k=1}^{K} \log(E_k) \cdot \cos\!\left(\pi n \cdot \frac{2k - 1}{2K}\right), \qquad n = 0, \ldots, N-1$$

The result is $N$ cepstral coefficients. The zeroth coefficient ($c_0$) captures overall energy; it is sometimes discarded or kept depending on the application.

### Comparison with MFCC Formula

MFCC applies the same DCT but over a Mel-warped filterbank where:

$$f_k^{\text{Mel}} = 700 \left( 10^{m_k / 2595} - 1 \right), \qquad m_k = m_{\min} + k \cdot \frac{m_{\max} - m_{\min}}{K+1}$$

The computational cost is identical — the difference is only in filterbank center placement.

## How It Connects to the nn Library

In the thesis pipeline, LFCC features feed into paraconsistent analysis for feature-set quality evaluation:

```
Signal (8 kHz) → STFT frames → LFCC filterbank → log energy → DCT → feature vectors
                                                                    → paraconsistent::evaluate()
```

The filterbank is implemented as part of the wave preprocessing utilities:

```
include/wave/lfcc_pipeline_utils.hpp
src/core/wave/                         # implementation
```

### Integration Example

```cpp
#include "wave/lfcc_pipeline_utils.hpp"
#include "paraconsistent/paraconsistent.hpp"

// Extract LFCC feature vectors from a signal
const int K = 24;   // number of filters
const int N = 13;   // number of cepstral coefficients
const int f_min = 0, f_max = 4000;  // Hz

auto frames    = nn::wave::frame(signal, /*window=*/512, /*hop=*/256);
auto lfcc_vecs = nn::wave::lfcc(frames, K, N, f_min, f_max, sample_rate);

// Evaluate quality of these features for speaker classes
auto result = nn::paraconsistent::evaluate(lfcc_vecs_by_speaker);
std::printf("LFCC D_truth = %.4f\n", result.D_truth);
```

## Common Pitfalls

1. **Low $K$ loses spectral resolution.** Use $K \geq 20$ filters for speaker verification tasks.

2. **Pre-emphasis is optional but common.** Applying $x'[n] = x[n] - 0.97 x[n-1]$ before framing boosts high-frequency energy and compensates for the roll-off of the vocal tract radiation model.

3. **Frame length matters.** At 8 kHz, 512 samples ≈ 64 ms gives good frequency resolution. Shorter frames trade frequency resolution for temporal resolution.

4. **Do not confuse LFCC with LPC.** Linear Predictive Coding (LPC) coefficients are different; LFCC refers specifically to the filterbank-based linear-spacing approach.

---

## See Also

- [Core/Paraconsistent](../Core/Paraconsistent.md) — Feature quality evaluation
- [Core/Wave](../Core/Wave.md) — Signal preprocessing and WAV loading
- [Core/Wavelet](../Core/Wavelet.md) — DTWPT alternative to filterbank features
- [Concepts/Imagined-Speech-and-EEG](./Imagined-Speech-and-EEG.md) — EEG companion signal
- [Research-Context](../Research-Context.md) — Thesis overview

---

## References

[1] C. Hanilçi, "Linear prediction residual features for automatic speaker verification anti-spoofing," *Multimedia Tools and Applications*, vol. 77, no. 13, pp. 16099–16111, Jul. 2018. DOI: 10.1007/s11042-017-5181-0.

[2] Y. M. Ali et al., "Speech-based gender recognition using linear prediction and mel-frequency cepstral coefficients," *Indonesian J. Electric Eng. Comput. Sci.*, vol. 28, no. 2, pp. 753–761, 2022.

[3] S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 28, no. 4, pp. 357–366, 1980.

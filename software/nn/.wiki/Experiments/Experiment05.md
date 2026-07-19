# Experiment05: Biometric Authentication of Severely Dysphonic Speakers via Imagined Speech

> **Thesis primary experiment.**  
> Thesis: *Autenticação Biométrica de Locutores Drasticamente Disfônicos Aprimorada pela Imagined Speech*  
> Author: André Furlan — UNESP | Advisor: Prof. Dr. Rodrigo Capobianco Guido | Funding: FAPESP 2021/12407-4

---

## Overview

Experiment05 implements the full speaker-authentication pipeline for individuals with severe laryngeal dysphonia (DLS). The core hypothesis: combining a degraded phonated voice with EEG-captured imagined speech produces a biometric signal that is both robust (works even when voice quality degrades) and difficult to spoof (EEG is not externally recordable without consent).

The experiment consists of two stages:

- **E3 — Feature Extraction**: handcrafted (DTWPT + classical descriptors, guided by paraconsistent quality ranking) vs. learned (SNN-AE, ANN-AE)
- **E4 — Authentication**: RNN or DSNN classifier, text-dependent and text-independent modes, nested or flat 5-fold cross-validation

> **Status (2026-07-19): both phases have completed a full run** — 208/208 Phase 00
> profiles and 32/32 Phase 01 profiles, 0 failures. The Phase-00 winner is the
> **same handcrafted extractor for both signals** — Haar wavelet, linear scale
> (LFCC), Category 1 — beating every one of the 24 autoencoder variants by a wide
> margin on `D_penalized` in both modalities. Phase 01's best configuration
> (early fusion, text-dependent, flat CV, unstandardized features) reaches
> **EER = 0.4534, AUC = 0.5452** — close to the chance level of a random
> classifier (EER=AUC=0.5), so authentication performance under the evaluated
> conditions does not yet support practical use. Full tables and discussion are
> in the thesis (`documentation/00-thesis/monography/chapters/09-testsAndResults.tex`,
> §Fase 00/§Fase 01/§Retomada das questões de pesquisa) and its Conclusões chapter.

---

## Theoretical Background

### Problem: Voice-Based Authentication Under Dysphonia

Conventional speaker-verification systems assume a clean, periodic voice signal. In severe laryngeal dysphonia, the phonation mechanism is impaired: aperiodic vibration, breathiness, and noise dominate the signal, obscuring the formant structure that encodes speaker identity.

Imagined (covert) speech is proposed here as a complementary signal: the speaker mentally rehearses an utterance without producing motor output. The perisylvian language network (Broca + Wernicke areas) activates and EEG captures neural correlates that do not depend on laryngeal phonation, so imagined-speech EEG remains available even when the voice signal is degraded.

### Paraconsistent Feature Selection (EPC/α/β)

Before training any classifier, the pipeline ranks all feature-strategy×modality combinations using the paraconsistent quality metric. This avoids expensive hyperparameter sweeps.

Features are scaled **per dimension to [0,1] across all samples** before scoring
(audit M-1), so every descriptor is commensurable and bounded. Then:

$$\alpha = \min_c \left( \frac{1}{F} \sum_{d=1}^{F} \big(1 - (\max_{c,d} - \min_{c,d})\big) \right) \quad \text{(intraclass similarity)}$$

where $\max_{c,d}, \min_{c,d}$ are the per-dimension extrema within class $c$ over its $M$ feature vectors.

$$\beta = \frac{R}{N\,(N-1)\,M\,F} \quad \text{(interclass overlap)}$$

where $R$ counts — over all ordered class pairs — how many feature-vector components fall inside another class's per-dimension $[\min, \max]$ range; $N$ classes, $M$ vectors per class, $F$ dimensions.

Map to paraconsistent plane:

$$G_1 = \alpha - \beta \qquad G_2 = \alpha + \beta - 1$$

$$D_{\text{truth}} = \sqrt{(G_1 - 1)^2 + G_2^2}$$

Smaller $D_{\text{truth}}$ → features closer to the "Truth" corner → better speaker separability before any classifier sees them. The combination with the minimum $D_{\text{truth}}$ is passed to Stage E4.

See [Paraconsistent Feature Engineering](../Core/Paraconsistent.md) for the full derivation and API.

### Handcrafted Feature Extraction (DTWPT-based)

The Discrete-Time Wavelet Packet Transform decomposes the signal into a full binary tree of sub-bands. For a signal of length N with a J-level DWPT:

$$x[n] = \sum_{j,k} c_{j,k} \psi_{j,k}[n]$$

The **mother wavelet** $\psi$ is a Phase-00 sweep axis (`handcrafted.wavelet`): any tag with coefficient traits in `include/wavelet/Types.hpp` — `haar` or `daubN` for even N in [4, 46] (23 in total). Shorter filters (e.g. Haar) localise sharp transients; longer Daubechies filters give smoother, more selective sub-bands. The best wavelet per signal is chosen by the paraconsistent ranking.

At each leaf node (sub-band), the following descriptors are computed:

| Descriptor | Formula | Sensitivity |
|---|---|---|
| Sub-band energy | $E_k = \sum_n c_k[n]^2$ | Overall power per band |
| ZCR | $\frac{1}{N}\sum_n \mathbb{1}[\text{sign}(x[n]) \neq \text{sign}(x[n-1])]$ | Voicing / noisiness |
| Entropy | $-\sum_k p_k \log p_k$ | Spectral spread |
| Teager–Kaiser | $\Psi[x(n)] = x^2(n) - x(n+1)x(n-1)$ | Amplitude modulation |
| Jitter | $\frac{\overline{|T_i - T_{i+1}|}}{\bar{T}}$ | Pitch period irregularity |
| Shimmer | $\frac{\overline{|A_i - A_{i+1}|}}{\bar{A}}$ | Amplitude irregularity |

Frequency scales evaluated: **BARK**, **MEL**, **LFCC** (see [LFCC](../Concepts/LFCC.md)).

> **The `scale` axis is voice-only (fixme.md D6).** Bark and Mel are **cochlear** scales — they model the frequency resolution of human *hearing*. There is no physiological basis for applying them to EEG, which is not sound, so `E05Config::validate()` rejects `scale != lfcc` when `modality=eeg`.
>
> They were also provably inert there, which is what exposed the problem. `group_by_scale()` normalizes the perceptual curve by the signal's **own Nyquist**. Bark spans ~24 Barks over the audible range and `n_bands=24`, so for voice the factor is ~0.97 — a no-op, the bin *is* the Bark number, and the curve's compression at high frequency genuinely merges sub-bands (bark→9 groups, mel→11, vs lfcc's 16). For EEG (Nyquist 512 Hz) the factor is ~4.96: the curve is stretched 5× to fill 24 bins. Bark/Mel are ~linear over that range, so the mapping becomes injective — each sub-band lands in its own bin and the grouping degenerates to *exactly* lfcc's one-group-per-sub-band. "Bark" for EEG was never Bark; it was a linearly rescaled pseudo-scale identical to linear. Verified on the stored results: all three scales gave bit-identical `D_truth` in **46/46** EEG wavelet×category groups (the only apparent exceptions, `daub32`, differed by ~1e-6 — float noise from that profile's individual re-run).
>
> Consequence worth knowing: because the three tied exactly, the EEG Phase-00 winner was decided by the ranking's **sort tie-break** and came out labelled `bark`. Reporting "the Bark scale won for EEG" would have been false — Bark did nothing, and the winning vector is literally the linear one. `01_e05_phase00_rank.py` now detects and reports exact ties so an arbitrary tie-break can't be mistaken for a result. Guarded by `E05EegScaleAxis.BarkAndMelAreRejectedForEeg`.

**Category 1 vs Category 2** (`handcrafted.cepstral`). With `cepstral=false` the per-band energies are the features (Category 1: linear/Mel/Bark-band energies). With `cepstral=true` a **log + DCT-II** is applied over the band energies, yielding cepstral coefficients — **LFCC / MFCC / BFCC** for `scale = lfcc / mel / bark` (Category 2). The cepstral coefficients replace the raw energy descriptor; other descriptors (ZCR, entropy, …) are still appended per band.

### Learned Feature Extraction (Autoencoders)

> **Scope note.** The **thesis** Phase 00 compares three feature-extraction routes through the paraconsistent ranking: **handcrafted**, **SNN-AE** (spiking autoencoder, `ProtocolSpikingAutoencoder`), and **ANN-AE** (non-spiking dense autoencoder, `ProtocolAutoencoder`). Both AE families are wired into the Experiment05 executable and shipped as Phase 00 profiles. The **LSTM-AE** remains in the code (built for the Guayaquil congress paper) but no thesis profile uses it.

**SNN-AE (`ProtocolSpikingAutoencoder`, implemented)**: spiking autoencoder. Encoder `Linear → LIF`, decoder `Linear → LIF-integrator`. The raw signal is average-pooled to 256 bins and **min-max normalized to `[0,1]`**.

*Temporal coding (required for a meaningful SNN).* A LIF autoencoder only carries information through spikes, so each normalized sample is expanded into `time_steps` (default 16) spike frames using `autoencoder.encoding`:
- `poisson` — each entry spikes with Bernoulli probability = its value (rate code);
- `latency` — stronger inputs fire earlier (`t_spike = round((1−v)(T−1))`, matches the Experiment04 encoder);
- `direct` — analog pass-through, no spikes (used by ANN-AE; also an SNN fallback).

The AE is trained (batched, MSE reconstruction) to reconstruct the spike frames. At readout the membrane state is **reset once per sample** and then **integrates across the T frames** (no reset between frames) — the temporal integration that lets weak, sub-threshold per-step current accumulate into spikes. The **per-sample feature is the mean latent over the T frames**.

Two knobs make this work and both are profile-configurable:
- `autoencoder.time_steps` (default 16) — integration window.
- `autoencoder.voltage_threshold` (default **0.2**, vs the LIF default 1.0) — the encoder LIF threshold. Normalized spike frames are low-amplitude, so at `V_th=1` no encoder neuron fires and the latent collapses to all-zeros. That collapse (plus the old single-analog-vector, one-forward presentation) is why the raw-vector SNN-AE ranked poorly. (Audit m-2: this note previously glossed `α≈0.5` as "coin-flip separability" — that reading came from the inverted convention. Under the correct one, `α≈0.5` is mid-range, and the best-ranked configs of the whole grid sit at `α≈0.64–0.67`; see the corrected table below.)

Spike frames are seeded from `experiment.seed` for reproducibility. Guarded by `E05SnnAe.*` tests (latent non-degenerate + varies across samples; encoding changes the feature).

> **Dead-latent guard: encoder firing-rate regularization (2026-07-16).** A collapsed encoder (constant output regardless of input) lands exactly on the paraconsistent *Ambiguity* vertex (`α=β=1`) and used to score deceptively well under `D_truth` alone — see [Core/Paraconsistent.md](../Core/Paraconsistent.md) for the `D_penalized` metric that closed that ranking exploit, and [Spike-Rate-Regularization](../Concepts/Spike-Rate-Regularization.md) for why a starved LIF encoder collapses in the first place (root cause: at He/Kaiming init the single spiking stage of `linear:16:leaky, linear:8:identity` fires at only ~8.3%, and the `identity` second stage adds no further nonlinearity, so a starved first stage starves the whole latent). `feature_extraction.autoencoder.{firing_rate_reg_lambda, firing_rate_min, firing_rate_max}` push each encoder `Lif` layer's mean firing rate into `[firing_rate_min, firing_rate_max]`, mirroring the DSNN classifier's mechanism (same math, see below) but implemented separately in `ProtocolSpikingAutoencoder::backward` — the AE's encoder is a generically-built `Sequential` (from `encoder_layer_spec`), not the classifier's named layers, so the Lif layers are located once at construction via `dynamic_cast`, and the regularization gradient is injected by a manual reversed-order backward loop rather than `Sequential::backward`. All 18 real `snn-ae` Phase 00 profiles (and their 18 `smoke` mirrors) now set `firing_rate_reg_lambda=0.5`, `firing_rate_min=0.10`, `firing_rate_max=0.80` — `0.10` sits above the measured native init rate (8.3%) so the penalty actually engages; `0.5` was validated on `direct` at the profile's real learning rate (baseline `α=0.5` → `α=0.375` with regularization on). Default is `0.0` (inert) when unset. Guarded by `Experiment03RedesignTest.ProtocolSnnFiringRateRegularization*` (inject-when-enabled / inert-when-zero).
>
> The 300 stored Phase 00 results predate this change and do not yet reflect it — re-running the 18 `snn-ae` profiles to regenerate results under the new default is a real cost (the `poisson`/`latency` encodings, T=16, took over 2h for a single run in testing) and has not been done.

**Encoding sweep (thesis comparison).** Phase 00 ships all three SNN-AE encodings — `poisson`, `latency`, `direct` — × the 3 compact sizes × 2 sources = 18 profiles (`p00_ae_snn_<encoding>_<size>_<source>.json`), so the thesis can directly compare rate coding, first-spike-time coding, and the (expected-degenerate) non-temporal baseline through the same paraconsistent ranking.

All 18 profile names (`profiles/phase00/`):

| encoding | tiny | small | base |
|---|---|---|---|
| `poisson` | `p00_ae_snn_poisson_tiny_eeg.json`, `..._voice.json` | `p00_ae_snn_poisson_small_eeg.json`, `..._voice.json` | `p00_ae_snn_poisson_base_eeg.json`, `..._voice.json` |
| `latency` | `p00_ae_snn_latency_tiny_eeg.json`, `..._voice.json` | `p00_ae_snn_latency_small_eeg.json`, `..._voice.json` | `p00_ae_snn_latency_base_eeg.json`, `..._voice.json` |
| `direct`  | `p00_ae_snn_direct_tiny_eeg.json`, `..._voice.json`  | `p00_ae_snn_direct_small_eeg.json`, `..._voice.json`  | `p00_ae_snn_direct_base_eeg.json`, `..._voice.json`  |

Each profile is identical except `feature_extraction.autoencoder.{encoding, time_steps, voltage_threshold}` and `encoder_layer_spec`/`decoder_layer_spec` widths (per size, see table below). `direct` profiles use `time_steps: 1` (ignored by the encoder anyway — a single analog frame) and `voltage_threshold: 1.0` (the LIF default — appropriate since `direct` is meant to reproduce the *un-encoded* baseline, not a tuned spiking regime); `poisson`/`latency` use `time_steps: 16`, `voltage_threshold: 0.2`.

> ### ⚠️ Encoding comparison: withdrawn pending re-run (fixme.md D2/F4)
>
> A table used to sit here reporting measured `α`/`D_truth` per SNN-AE encoding (`direct` best,
> `poisson` last) over the Phase 00 grid. **It has been removed rather than corrected**, because
> every one of its numbers is unsupported and none can be recomputed:
>
> 1. **Produced under the `snn_lr_scale` bug (D3).** Every AE weight trained at an effective
>    lr of 1e-4 while the profiles declared 1e-3. The numbers describe a run nobody configured.
> 2. **Its ranks are arithmetically impossible now.** It said "rank (of 150, eeg)"; after D6
>    retired the degenerate EEG bark/mel axis, EEG has **58** combinations (46 handcrafted + 12 AE).
> 3. **The source data is gone.** All Phase 00 results were deleted (see `results/thesis/phase00/README.md`);
>    the re-run is postponed, so the table cannot be regenerated.
>
> Correcting the figures was not an option — there are no figures to correct. Restating them with a
> caveat would have kept three-decimal precision on numbers with no run behind them, which is how
> the *previous* revision of this table (a 20-epoch/2-fold preliminary run, reported as if final and
> with the direction backwards) survived long enough to reach the thesis.
>
> **What still stands** is the mechanism, which is theory rather than measurement, and which
> predicts the ordering independently of any run:
>
> - `α` is built from the intraclass **min–max range**, so it is maximally outlier-sensitive: a few
>   extreme samples per class widen the range to the whole normalised interval and drive `α→0`.
> - Each encoding injects a different, *quantifiable* amount of noise **before any learning happens**.
>   `poisson` redraws an independent Bernoulli per time step, so the mean over `T=16` frames carries
>   variance `v(1-v)/T` — at `v=0.5` that is a standard deviation of **≈0.125, i.e. ~12.5% of the
>   feature's full [0,1] range**, added per sample regardless of how well the encoder trained.
>   `latency` is deterministic: the same `v` always yields the same spike time, and its quantisation
>   error is at most `½·1/(T−1) ≈ 0.033`. `direct` injects none.
> - That ladder (none / small-deterministic / large-random) predicts the observed ordering, and
>   predicts it as an artifact of the **metric plus the encoder's noise floor**, not as evidence that
>   temporal codes carry less speaker information.
>
> So the earlier framing — "a negative result for temporal coding" — **is not established**, and is
> not merely uncertain: `α` cannot separate "the encoder failed to learn" from "this encoding has an
> irreducible statistical floor at T=16". Since the floor scales as `1/T`, the claim is testable by
> re-running `poisson` at larger `T` (T=64 shrinks the variance 4×). That test has **not** been run.
> See fixme.md **D2** for the decision and the derivation.

**ANN-AE (`ProtocolAutoencoder`, implemented)**: non-spiking dense autoencoder — same flat 256-dim pooled input and 2:1 compression, ReLU activations. Serves as the non-spiking baseline against SNN-AE.

**LSTM-AE (legacy, Guayaquil paper — not in the thesis Phase 00 grid)**: sequence-to-sequence autoencoder. Encoder LSTM processes windowed frames, final hidden state = latent vector; decoder LSTM reconstructs the frame sequence. Trained with MSE + BPTT. See [LSTM and BPTT](../Concepts/LSTM-and-BPTT.md).

> **Windowing + batching (AE-on-EEG fix).** The raw signal is *framed* into at most `kAeMaxFrames` (64) windows of `frame_len` samples each → AE input `(T_frames, frame_len)` with `input_size = frame_len`, `seq_len ≤ 64`. This replaced the earlier `input_size=1, seq_len=24576` wiring, which fed the whole flattened multi-channel EEG as one length-24576 sequence — both semantically wrong and far too long to unroll (and it crashed once the `LSTMAutoencoder` met the trainer's batched 3-D tensor). `LSTMAutoencoder` now handles both 2-D `(T,D)` and 3-D `(B,T,D)` inputs (`LSTMLayer` already did the batched BPTT; the projections/last-step/replicate were made batch-aware). Verified against snnTorch/PyTorch — see [Ground-Truth and Smoke Testing](../Guides/Ground-Truth-and-Smoke-Testing.md).

**Compact-AE capacity sweep (low-power design).** Phase 00 ships two AE families (SNN-AE, ANN-AE) × three compact sizes per signal, all single-layer (shallow, edge-friendly) with a 2:1 hidden→latent compression ratio:

| size | hidden | latent (= feature-vector dim) |
|---|---|---|
| `tiny`  | 16 | 8  |
| `small` | 32 | 16 |
| `base`  | 64 | 32 |

The latent layer *is* the feature vector, so a smaller bottleneck yields a smaller vector — cheaper downstream classifier and often better generalisation. These choices follow the lightweight-autoencoder-for-edge literature (aggressive bottleneck, ~2:1 compression, shallow stack). The wired families are `snn-ae` and `ann-ae` (each swept over the three sizes); `lstm-ae` is accepted by the config but no thesis profile ships it. See [Autoencoders](../Concepts/Autoencoders.md) and [Memory-Constrained Design](../Guides/LSTM-Performance.md).

Autoencoder training in Experiment05 is unsupervised (no speaker labels), and latent vectors feed paraconsistent ranking and downstream classifier.

### Authentication: Residual Network (RNN) and Deep SNN (DSNN)

> **Scope note.** The non-spiking **RNN** classifier was built for the Guayaquil congress paper. The **thesis** uses only the spiking classifier (**DSNN**). The RNN is documented here because both share the Experiment05 code path.

**RNN** (here: Residual Neural Network, not recurrent):  
Skip connections prevent vanishing gradients in deep classifiers. Residual block:

$$\text{out} = F(x) + x$$

where $F$ = 2 Linear layers with BatchNorm + ReLU. Output layer: `linear:N_speakers:identity` → cross-entropy loss. Implemented in `SimpleResNet.hpp`.

**DSNN**: temporal deep **residual** spiking classifier (audit M-4). Each hidden block (`Linear → (tdBN) → LIF`, all `hidden_dim` wide) is wrapped in an identity skip connection `h_out = block(h) + h`; `residual:D` in the layer spec sets the number of such blocks. Skips are parameter-free (no added serialization state) and sit only around the equal-width hidden blocks — the input stage (`input_dim → hidden_dim`) and output stage carry none. This matches the deep-residual-SNN design that tdBN was introduced to train (Zheng et al., AAAI 2021). The static feature vector is **rate-encoded by constant-current injection over `kSnnTimeSteps` (default 16) time steps**; `LifBPTT` neurons integrate the resulting current over time (time-major `(T*B, F)` layout) through a `Linear → LIF → (Linear → LIF)* → Linear` stack. The readout is the **mean firing rate (spike-count / T)** of the final layer, yielding class logits; gradients flow via full backpropagation-through-time with surrogate spike derivatives. This is a genuine temporal SNN, not a single-step thresholding net. See [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md) and [Layers](../Core/Layers.md).

### Regularization & Normalization

Three techniques guard generalisation / trainability on the small dysphonic-speaker dataset, all off by default and enabled per profile:

- **Decoupled L2 weight decay** (`training.weight_decay`, AdamW): applies to **both** the RNN and DSNN classifiers. Shrinks only 2-D weight matrices; biases and the SNN biophysical scalars (R, C, V_th) are excluded so `τ = R·C` and the firing threshold are never decayed. See [Optimizers](../Core/Optimizers.md#decoupled-weight-decay-adamw--sgdw).
- **Firing-rate regularization** (`training.firing_rate_reg_lambda` for the DSNN classifier; `feature_extraction.autoencoder.firing_rate_reg_lambda` for the SNN-AE encoder): pushes each spiking layer's mean firing rate into the band `[firing_rate_min, firing_rate_max]` (default `[0.05, 0.80]`), preventing **dead neurons** (rate → 0, the surrogate gradient vanishes) and **bursting neurons** (rate → 1, selectivity lost). The penalty $\lambda\sum_\text{layers}\big(\max(0, r_\text{min}-r)^2 + \max(0, r-r_\text{max})^2\big)$ is differentiated to `2λ(r − clamp(r, r_min, r_max))/n` and injected into the incoming gradient at each LIF spike output during backward — the same math as `SpikeCountLossImpl`, but two independent gradient-injection implementations: `E05DsnnClassifier::add_firing_rate_grad` (classifier, named layers) and `ProtocolSpikingAutoencoder`'s `backward_with_firing_rate_reg` (AE encoder, `Sequential`-based, Lif layers located via `dynamic_cast`; see the SNN-AE section above). Inert when `λ = 0` (the RNN classifier and the ANN-AE have no spiking layers to regularize either way).
- **Threshold-Dependent Batch Normalization** (`training.batch_normalization = "threshold-dependent"`, DSNN-only): inserts a tdBN layer after each `Linear` and before each `LifBPTT` (`fc_in → tdBN → lif_in → (hidden_fc → tdBN → hidden_lif)* → fc_out`). It normalizes the pre-spike current over batch+time and rescales it to `N(0,(α·V_th)²)` (α = `training.tdbn_alpha`, default 1), stabilizing deep-SNN training and guarding against the No-Spike Problem. See [Threshold-Dependent Batch Normalization](../Concepts/Threshold-Dependent-Batch-Normalization.md). Inert for the RNN classifier.

**Input feature standardization** (`training.standardize_features`, default **on**): before the classifier, each feature dimension is z-scored. The mean/std are fit on the **training rows of each fold only** and applied to train and test, so test statistics never leak into the scaler (`fit_scaler`/`apply_scaler` in `E05Classifiers.cpp`). This is the CMVN-style input normalization + leakage guard described in [Data Normalisation](../Concepts/Data-Normalisation.md); unlike the three techniques above it is a preprocessing step, not a classifier-only regularizer. Constant dimensions (std ≈ 0) map to 0.

### Text-Dependent vs Text-Independent Evaluation

| Mode | Train phrases | Test phrases | Difficulty |
|---|---|---|---|
| Text-dependent | fixed set | same fixed set | Easier; high overlap |
| Text-independent | arbitrary | different arbitrary | Harder; tests generalisation |

Both modes use the same architecture; only the data split changes. Text-independent is the primary scientific contribution mode.

### Nested 5-Fold Cross-Validation

To avoid optimistic bias from hyperparameter tuning on the test fold, a nested k-fold scheme is used when `training.nested_cv=true`:

- **Outer loop** (5 folds): hold out one fold as test set; report final metrics here
- **Inner loop** (5 folds within training set): model selection by inner validation accuracy

When `training.nested_cv=false`, flat grouped k-fold is used.

See [K-Fold Cross-Validation](../Concepts/K-Fold-Cross-Validation.md).

---

## Dataset

**10.1117/12.2255697** — EEG imagined speech (public):
- 15 Spanish-speaking subjects
- Utterances: vowels (`/a/ /e/ /i/ /o/ /u/`) + 6 directional commands (`arriba/abajo/adelante/atras/derecha/izquierda`)
- Two speech conditions: **pronounced** (audio + EEG recorded simultaneously) and **imagined** (EEG only — no audio was recorded in the imagined condition)
- Audio: 44100 Hz, single channel (present only for pronounced trials)
- EEG: 1024 Hz, 6 channels (F3, F4, C3, C4, P3, P4), 10-20 system; each 4 s trial = 4096 samples/channel

> The pipeline's `modality` field (`voice` | `eeg` | `fused`) selects which signal(s) feed feature extraction — it is not a dataset-level modality. Fusion is not a third recorded signal; it combines the two existing signals, and *when* the combination happens is a second axis, `fusion_mode` (only consulted for `modality=fused`):
>
> | `fusion_mode` | Where combination happens | Extractor runs | Result vector |
> |---|---|---|---|
> | `early` | Raw voice+EEG samples concatenated **before** extraction | one pass over the joined signal | features of the joined signal |
> | `late` (default) | Voice and EEG extracted **independently**, feature vectors concatenated **after** | two passes (one per signal) | `[voice-features ‖ eeg-features]` |
>
> Early fusion lets the extractor model cross-modal interactions but forces one sample rate onto both halves (the voice rate, since audio dominates the sample count); late fusion keeps each signal at its native rate and its own extractor, at the cost of never seeing the two jointly. Both are worth comparing as an experimental axis. Implemented in `E05FeatureExtraction.cpp::extract_features` (audit C12).

See [Data Loaders](../Core/DataLoaders.md) for the `E05Dataset` loader API and file layout.

---

## Implementation

### Module plan

```
src/experiments/05/
├── experiment05.cpp              CLI entry point
├── lib/
│   ├── include/
│   │   ├── E05Config.hpp         profile JSON parser
│   │   ├── E05Dataset.hpp        EEG + voice loader (wraps 10.1117 loaders)
│   │   ├── E05FeatureExtraction.hpp  handcrafted + autoencoder pipeline
│   │   ├── E05Paraconsistent.hpp paraconsistent ranking step
│   │   ├── E05Classifiers.hpp    RNN / DSNN authentication
│   │   └── E05Output.hpp         CSV, JSON, DAT writers
│   └── src/
│       ├── E05Config.cpp
│       ├── E05Dataset.cpp
│       ├── E05FeatureExtraction.cpp
│       ├── E05Paraconsistent.cpp
│       ├── E05Classifiers.cpp
│       └── E05Output.cpp
├── profiles/
│   ├── debug.json                              (RNN quick smoke)
│   ├── phase00/  Phase 00 — feature construction + paraconsistent ranking (classifier.enabled=false)
│   │   ├── p00_hc_<wavelet>_<scale>_<cat>_<source>.json  wavelet(23) ×
│   │   │       cat ∈ {c1=energy, c2=cepstral LFCC/MFCC/BFCC} ×
│   │   │       scale: voice = bark/mel/lfcc (138); eeg = lfcc only (46)  = 184
│   │   │       (eeg drops bark/mel: cochlear scales, no basis for EEG — see D6 note below)
│   │   ├── p00_ae_ann_<size>_<source>.json           ANN-AE, size ∈ {tiny,small,base}
│   │   │                                              (latent 8/16/32, 2:1 hidden) × source(2) = 6
│   │   └── p00_ae_snn_<encoding>_<size>_<source>.json SNN-AE, encoding ∈ {poisson,latency,direct}
│   │                                                × size ∈ {tiny,small,base} × source(2) = 18
│   │                                                (6 ann + 18 snn = 24 AE)   → 208 total
│   └── phase01/  Phase 01 — DSNN authentication, best combo only (classifier.enabled=true)
│       └── p01_dsnn_<source>_<text>_<cv>_<std>.json   source(4) × text(dep/indep) ×
│           cv(nested/flat) × std ∈ {std,raw} (standardize_features ablation) = 32
│           (feature_extraction = placeholder; set the Phase-00 winner before running)
  └── tests/
  ├── e05_profile_audit_gtest.cpp        profile schema and validate() audit (all official profiles)
  ├── e05_feature_extraction_gtest.cpp   descriptor functions + extract_handcrafted
  └── e05_classifiers_gtest.cpp          batch evaluate() + aggregate stats + DSNN smoke path
```

### Profile schema

```jsonc
{
  "experiment": {
    "run_tag": "e05_handcrafted_eeg",
    "seed": 42,
    "repeats": 3,
    "seed_deterministic": false
  },
  "dataset": {
    "root": "/path/to/10.1117/",
    "results_dir": "results/",
    "modality": "eeg",         // "voice" | "eeg" | "fused"
    "fusion_mode": "late"      // "early" | "late"  — only used when modality=fused
  },
  "feature_extraction": {
    "strategy": "handcrafted",  // "handcrafted" | "autoencoder"
    "handcrafted": {
      "transform": "dtwpt",     // only value accepted in current Experiment05 baseline
      "scale": "lfcc",          // "bark" | "mel" | "lfcc"
      "descriptors": ["energy", "zcr", "entropy", "teager", "jitter", "shimmer"]
    },
    "autoencoder": {
      "model": "snn-ae",        // snn-ae | ann-ae (thesis Phase 00); lstm-ae accepted but unused
      "encoder_layer_spec": ["linear:64:leaky", "linear:32:identity"],
      "decoder_layer_spec": ["linear:64:leaky", "linear:output:identity"],
      "encoding": "poisson",    // snn-ae temporal code: poisson | latency | direct
      "time_steps": 16,         // snn-ae integration window
      "voltage_threshold": 0.2, // snn-ae encoder LIF threshold (below LIF default 1.0)
      "firing_rate_reg_lambda": 0.5,  // snn-ae encoder band penalty; 0 = off (default)
      "firing_rate_min": 0.10,
      "firing_rate_max": 0.80
    }
  },
  "numerics": {
    // false = FastActivations' softsign gates (~2x faster LSTM, provably NOT torch:
    // |tanh - tanh_fast| reaches 0.306). PyTorch/snnTorch is this project's correctness
    // reference, so exact is the default and the approximation is an explicit opt-in.
    "exact_activations": true
  },
  "paraconsistent": {
    "enabled": true
  },
  "classifier": {
    "type": "rnn",              // "rnn" | "dsnn"
    "layer_spec": ["linear:128:relu", "residual:2", "linear:N_speakers:identity"],
    "text_mode": "dependent"    // "dependent" | "independent"
  },
  "training": {
    "epochs": 50,
    "learning_rate": 1e-3,
    "samples_per_batch": 32,
    "early_stop_patience": 10,
    "k_folds": 5,
    "nested_cv": true,
    "optimizer_type": "adam",        // adam | sgd | lion | schedule-free-adamw
    "optimizer_momentum": 0.0,       // sgd only
    "gradient_clip_norm": 0.0,       // 0 = OFF (default), matching PyTorch. The ONLY
                                     // clipping knob: MSELoss/MAELoss used to also clip
                                     // themselves at norm 1.0, silently, overriding this.
    // learning_rate is OPTIONAL: omit it and each optimizer gets its own reference
    // default (adam 1e-3, sgd 1e-2, lion 1e-4, schedule-free-adamw 2.5e-3). Set it
    // explicitly only to sweep. Either way the resolved value + its source land in
    // the run summary's "training" block. See Core/Optimizers.md.
    "weight_decay": 1e-4,            // decoupled L2 (rnn + dsnn); 0 = off
    "firing_rate_reg_lambda": 0.01,  // dsnn-only band penalty; 0 = off
    "firing_rate_min": 0.05,
    "firing_rate_max": 0.80,
    "batch_normalization": "threshold-dependent",  // dsnn-only tdBN; "none" = off
    "tdbn_alpha": 1.0                // α: target std = α·V_th
  }
}
```

### Progress bars (three levels)

The experiment shows three concurrent progress bars during a run:

| Bar | Created by | Tracks |
|---|---|---|
| `Feature extraction` | `extract_features()` | samples processed (OpenMP parallel) |
| `Fold N/K \| <label>` | `ProgressCallback` (per outer fold) | epoch + batch within that fold |
| `E05 \| <run_tag>` | `experiment05.cpp` main | outer folds completed across all feature sets |

All bars are rendered by `nn::progress::ProgressManager` (background thread, ANSI escape codes). The global bar is completed and `ProgressManager::shutdown()` called before any output is written.

### Pipeline flow

```
10.1117/12.2255697 dataset
  │
  ├── Pronounced speech (44100 Hz, single channel)
  │     └── preprocessing (normalization, pre-emphasis, windowing)
  │           ├── Handcrafted: DTWPT + ZCR + entropy + Teager + jitter/shimmer
  │           └── Learned: autoencoder latent vectors
  │
  └── Imagined speech EEG (1024 Hz, 6 ch)
        └── preprocessing (per-window z-score)
              ├── Handcrafted: DTWPT energy per EEG band (alpha/beta/theta)
              └── Learned: SNN-AE / ANN-AE latent vectors
                    │
                    ▼
            Paraconsistent evaluation (α/β → D_truth)
            Rank all (strategy × modality × scale) combinations
            Select best — no classifier trained yet
                    │
                    ▼
            Classifier (RNN or DSNN)
            Nested or flat 5-fold cross-validation
            Text-dependent + text-independent splits
                    │
                    ▼
            Results: accuracy, EER, D_truth per combination
```

### EER scoring strategies (`IEERScorer`)

EER computation is pluggable via `statistics::IEERScorer`:

```cpp
// Pass to run_classifier(); nullptr → GenuineImpostorEERScorer (default)
statistics::GenuineImpostorEERScorer sota_scorer(/*n_enroll=*/1);
auto result = e05::run_classifier(view, fvs, label, cfg, &sota_scorer);

// Back-compat alias — now delegates to the genuine/impostor method (audit m-4)
statistics::ClassificationEERScorer alias_scorer;
auto result2 = e05::run_classifier(view, fvs, label, cfg, &alias_scorer);
```

| Scorer | Protocol | EER source | SOTA? |
|---|---|---|---|
| `GenuineImpostorEERScorer` | enroll N utterances → cosine similarity genuine/impostor trials → FAR/FRR sweep | Score distribution | ✓ |
| `ClassificationEERScorer` | upgraded (audit m-4): delegates to genuine/impostor with 1 enrollment utterance | Score distribution | ✓ |

`GenuineImpostorEERScorer` protocol:
1. Per speaker: first `n_enroll` utterances → L2-normalised mean embedding (template)
2. Remaining utterances → probes
3. Each probe scored vs. all templates: cosine similarity
4. Genuine trials (own speaker) vs. impostor trials (other speakers) → sort → threshold sweep → interpolated crossing = EER

See `include/statistics/eer_scorer.hpp` for the `ISplitPolicy`-style interface; add new strategies by implementing `IEERScorer::compute_eer(embeddings, labels, n_classes)`.

### Performance notes

**Parallel feature extraction** — `extract_features()` uses `#pragma omp parallel for schedule(dynamic, 4)` over samples. The DTWPT computation (`wavelets::malat`) has no global state, so parallelism is safe. Pre-sized `fs.vectors.resize(n_samples)` avoids `push_back` races; progress counter uses `#pragma omp atomic capture`.

**Pre-built dataset tensors** — `run_classifier()` builds `(N, D)` input and `(N, C)` one-hot target tensors once via `mutable_data_ptr()` before the outer fold loop. Each fold slices row views via `Tensor::row(idx)` — cheap view, no copy.

**True batch training** — `Trainer::fit_loop_supervised` stacks a `(B, D)` batch each iteration, doing one GPU kernel per layer instead of B tiny `(1, D)` kernels. Both `LinearImpl` and `CrossEntropyLoss` support arbitrary batch dimension. See [Training](../Core/Training.md#true-batch-supervised-training).

**Batch evaluate()** — test-set evaluation stacks all test samples into one `(N_test, D)` tensor, one `model.forward()` call, then inline argmax + one-vs-rest confusion matrix.

---

## Build and run

```bash
# Configure (once)
cmake --preset=max-performance

# Build experiment binary
cmake --build out/build/max-performance --target experiment05 -j$(nproc)

# Phase 00 — rank feature extractors per signal (stops after paraconsistent ranking)
./out/build/max-performance/src/experiments/05/experiment05 \
  --config src/experiments/05/profiles/phase00/p00_hc_daub4_lfcc_voice.json

# Phase 01 — DSNN authentication with the chosen extractor
./out/build/max-performance/src/experiments/05/experiment05 \
  --config src/experiments/05/profiles/phase01/p01_dsnn_eeg_indep_nested.json

# Full article run
./scripts/pipeline/run_experiment05.sh
```

### Two-phase protocol

> **Running it:** step-by-step commands (build → phase00 → rank → apply → phase01, with the `run_e05_profiles.sh` runner and live progress) are in [Running Experiment05 Profiles](../Guides/Running-Experiment05-Profiles.md).

The experiment is split into two profile sets, gated by `classifier.enabled`:

- **Phase 00 — feature-vector construction** (`profiles/phase00/`, `classifier.enabled=false`, `paraconsistent.enabled=true`). For each signal (`voice`, `eeg`), sweep the handcrafted extractor over **mother wavelet** (`handcrafted.wavelet`, 23 options with coefficient traits in `include/wavelet/Types.hpp`: `haar` + `daub4`…`daub46`) × **category** (`c1` energy / `c2` cepstral) × **scale** — `bark`/`mel`/`lfcc` for voice (138) but **`lfcc` only for EEG** (46; see the scale note below) = 184, plus the **compact AE sweep** — ANN-AE × `tiny`/`small`/`base` (6) and SNN-AE × `poisson`/`latency`/`direct` encoding × `tiny`/`small`/`base` (18) = 24 per signal, for **208** rankings, and score every combination with the paraconsistent metric. The run stops after ranking — no classifier is trained, and `layer_spec` is not required. Output: paraconsistent CSV + summary JSON only. Pick the lowest-`D_truth` combination per signal; fused vectors are built afterward from each side's winner.
- **Phase 01 — authentication** (`profiles/phase01/`, `classifier.enabled=true`, `paraconsistent.enabled=false`). Feed **only the Phase-00 winning combination** into the DSNN and report EER/AUC. The `feature_extraction` block in these profiles is a placeholder (handcrafted / lfcc / daub4) — set the winning wavelet+scale (or `strategy=autoencoder`) before running. Crosses source (`voice`, `eeg`, `fused-early`, `fused-late`) × text mode × CV scheme × `standardize_features` (on/off ablation) = 32.

**Automating the Phase 00 → Phase 01 hand-off** — three scripts remove the manual seams, and
(since 2026-07-18) `run_e05_profiles.sh phase00`/`all` chains all three itself once every
phase00 profile passes (fails closed on any failure; `E05_FORCE_POST`/`E05_SKIP_POST`
override — see [Running Experiment05 Profiles](../Guides/Running-Experiment05-Profiles.md)
for the full gating rules and the `scope=all` ordering caveat). Shown here for manual re-runs:

```bash
# Rank Phase 00 results, pick the min-D_truth winner per signal:
python3 scripts/pipeline/e05/01_e05_phase00_rank.py \
  --profiles-dir src/experiments/05/profiles/phase00 \
  --results-dir  results/thesis/phase00 \
  --out          results/thesis/phase00/winners.json

# Inject the winners into the 32 Phase 01 profiles (fused uses the voice winner by default):
python3 scripts/pipeline/e05/02_e05_apply_winner.py \
  --winners      results/thesis/phase00/winners.json \
  --profiles-dir src/experiments/05/profiles/phase01

# Regenerate the thesis Phase 00 tables from the ranked results:
python3 scripts/pipeline/e05/e05_build_phase00_paraconsistent_tables.py \
  --results-dir results/thesis/phase00 \
  --tables-dir  ../../documentation/00-thesis/monography/tables
```

`01_e05_phase00_rank.py` collates every `results/thesis/phase00/*_paraconsistent.csv`, selects on `d_penalized` (not raw `D_truth`, which a collapsed latent can game) per signal, reports exact ties explicitly, and writes `winners.json` (each winner carries its full `feature_extraction` block). `02_e05_apply_winner.py` rewrites each Phase 01 profile's `feature_extraction` to its source's winner (`voice→voice`, `eeg→eeg`, `fused-*→--fused`); it is idempotent and supports `--dry-run`. `e05_build_phase00_paraconsistent_tables.py` regenerates the committed thesis tables — it defaults `--tables-dir` to that directory, so pointing it at anything but complete phase00 results overwrites what the thesis compiles from. It ranks **ascending** by `d_penalized` (rank 1 = best) and marks each signal's overall winner (chosen across both the handcrafted and autoencoder families) with a dagger — see pitfall 12 below for why this matters.

A fourth script, not yet chained into `run_e05_profiles.sh`, ranks the Phase 01 results once that phase has run:

```bash
# Regenerate the thesis Phase 01 authentication table (32 configs, ranked by mean EER):
python3 scripts/pipeline/e05/e05_build_phase01_auth_tables.py \
  --results-dir results/thesis/phase01 \
  --tables-dir  ../../documentation/00-thesis/monography/tables
```

`e05_build_phase01_auth_tables.py` averages `mean_eer`/`mean_auc` across each configuration's 3 repeats and writes `tables/phase01_auth.csv`, sorted ascending by EER (lower is better) — the source of the §Fase 01 table and the headline numbers in the status note above.

---

## Outputs

| File | Contents |
|---|---|
| `results/e05_*_metrics.csv` | Per-fold: accuracy, F1, precision, recall, EER, AUC, model_path |
| `results/e05_*_paraconsistent.csv` | α, β, G₁, G₂, D_truth per (strategy × modality × scale) |
| `results/e05_*_summary.json` | Config, seed, mean±std±ci95 for all metrics + per-fold model paths |
| `results/e05_*_comparison.dat` | pgfplots DAT: all aggregate metrics for thesis figures |
| `results/guayaquil/models/<run_tag>/<feature_label>/fold_N.bin` | Trained model state dict per outer fold (binary, `nn::io` format) |

### Data fed to each profile family, and its metadata

Every profile pulls from the **same** paired audio+EEG trial set (`load_dataset` drops any trial missing either signal, so all modality runs are comparable — see [Dataset](#dataset)). What differs per profile family is *which signal(s)*, *what preprocessing*, and *what tensor shape* reaches the extractor:

| Profile family | Input signal(s) | Preprocessing | Shape fed to the extractor | Sample rate used |
|---|---|---|---|---|
| Handcrafted (`p00_hc_*`) | `sample.audio` or `sample.eeg` per `dataset.modality` | Voice: pre-emphasis (`α=0.97`). EEG: none. Zero-padded to next power of two for DTWPT. | Flat 1-D signal (voice: 176400 samples; EEG: 6 ch × 4096 = 24576, **channel-major flattened** — all of ch0, then ch1, …) | Voice 44100 Hz, EEG 1024 Hz (nominal, per Pressel Coretto et al. 2017) |
| ANN-AE (`p00_ae_ann_*`) | same as handcrafted, single signal | Average-pooled to 256 bins, **no normalization** (ReLU tolerates raw scale) | `(1, 256)` per sample, one forward pass | n/a (pooled, rate-agnostic) |
| SNN-AE (`p00_ae_snn_*`) | same as handcrafted, single signal | Average-pooled to 256 bins, **min-max normalized to `[0,1]`**, then spike-encoded into `time_steps` frames (`poisson`/`latency`/`direct`) | `(1, 256)` **× T frames**, membrane state reset once per sample and integrated across all T frames | n/a (pooled) |
| Fused-early (`*-fused-early`) | `sample.audio` ++ `sample.eeg` concatenated | Voice preprocessing applied to the whole concatenation (approximation — documented in code, not a physical resample) | Flat 1-D, voice length + EEG length | 44100 Hz applied to the whole signal |
| Fused-late (`*-fused-late`) | both signals, extracted independently | Each signal's own preprocessing above | Two independent extractions, vectors concatenated **after** | each signal's own rate |
| Phase 01 (`p01_dsnn_*`) | the Phase-00 winner's vectors (already extracted) | z-score standardization (`training.standardize_features`), fit on train folds only | `(N_samples, feature_dim)` | n/a — operates on feature vectors, not raw signal |

**EEG flattening order matters**: `E05Sample::eeg` is `(N_channels=6, N_samples=4096)`; `tensor_to_vec` iterates rows-then-cols, so the flattened signal is channel-major (channel 0's full 4096-sample run, then channel 1's, …), **not** time-interleaved across channels. Anything reading a raw EEG feature vector must account for this layout.

**Per-sample dataset metadata** (`E05Sample`, from `load_dataset`): `subject_id` (int, groups outer folds — GroupKFold, never split across train/test), `stimulus` (int, 1–10; mapped to `text_phrase` via `stimulus_to_phrase`: vowels `a/e/i/o/u` for 1–5, directional words `arriba/abajo/izquierda/derecha/adelante` for 6–10), `text_phrase` (string, drives `classifier.text_mode` dependent/independent splitting). These are **not** written per-row into the paraconsistent/metrics CSVs (which are aggregate-only); they are consumed internally by the fold-splitting and text-mode logic.

**Run-level metadata written to `summary.json`** (self-describing result files — added so a run's config doesn't have to be cross-referenced against its source profile):
- Always: `run_tag`, `seed`, `modality`, `strategy`, `classifier`, `text_mode`.
- `dataset.{n_subjects, n_stimuli, n_samples}` — the **actual** composition used (post audio+EEG pairing drop, can be smaller than the raw `.mat` trial count).
- `strategy=="handcrafted"` → `handcrafted.{wavelet, scale, cepstral, dtwpt_level, descriptors}`.
- `strategy=="autoencoder"` → `autoencoder.{model, encoder_layer_spec, decoder_layer_spec}`, plus for `model=="snn-ae"`: `encoding`, `time_steps` (forced to `1` when `encoding=="direct"`, regardless of the profile's configured value — the summary records what actually ran, not the raw field), `voltage_threshold`.

This closes the gap where all 18 SNN-AE profiles previously shared the identical FeatureSet label `"autoencoder-snn"` in the paraconsistent CSV — indistinguishable except by output *filename*. The label still doesn't carry encoding/size (labels are shared across the sweep by design, since paraconsistent ranking compares FeatureSets by label), but `summary.json` now does. Guarded by `E05Output.*` gtests (handcrafted vs. autoencoder field presence, snn-ae `direct` forcing `time_steps=1`, ann-ae omitting snn-only fields, dataset composition roundtrip).

### Model checkpoints

After each outer fold, `run_classifier()` serializes the trained classifier state dict (RNN or DSNN) via `nn::io::save_state_dict(model.state_dict(), path)`. To reload a RNN checkpoint:

```cpp
#include "io/StateIO.hpp"
#include "layers/residual/SimpleResNet.hpp"

SimpleResNetImpl<nn::Backend> model(feat_dim, 128, n_speakers, 2);
auto sd = nn::io::load_state_dict("results/guayaquil/models/run/feat/fold_0.bin");
model.load_state_dict(sd);
```

Path pattern: `<results_dir>/models/<run_tag>/<feature_label>/fold_<N>.bin`

### Metric definitions

**Verification-only protocol (audit C-1).** Outer folds are speaker-disjoint
(GroupKFold by `subject_id`): test speakers are never seen in training. This is
an x-vector-style verification setup — train on background speakers, enrol the
unseen test speakers at evaluation time. Closed-set identification metrics
(accuracy / macro-F1 / precision / recall / specificity) are therefore **not
reported** (emitted as NaN); **EER and AUC are the primary metrics**.

| Metric | Formula | Notes |
|---|---|---|
| Accuracy / macro-F1 / P / R / specificity | — | **Not reported** (NaN) under verification-only: closed-set argmax over unseen speakers is invalid (audit C-1) |
| EER | FAR = FRR threshold crossing | **Primary.** `GenuineImpostorEERScorer`: cosine genuine/impostor trials |
| AUC | P(genuine_score > impostor_score) | **Primary.** Wilcoxon–Mann–Whitney; complement to EER |
| std_* | Sample std over folds | `sqrt(Σ(xi-μ)²/(n-1))` (audit M-3) |
| ci95_* | `t_{0.975,n-1} × std / √n` | Student-t CI for small fold counts (audit M-3) |

---

## Key differences from Experiment04

| Aspect | Experiment04 | Experiment05 |
|---|---|---|
| Purpose | Congress paper (SNN vs LSTM reconstruction) | Thesis primary experiment |
| Dataset | FSDD (spoken digits, 8 kHz, audio only) | 10.1117/12.2255697 (EEG 1024 Hz + voice 44100 Hz) |
| Task | Autoencoder reconstruction (MSE) | Speaker verification (EER, AUC; speaker-disjoint folds) |
| Signals | Audio only | Voice + EEG (bimodal) |
| Feature selection | Fixed architecture sweep | Paraconsistent α/β ranking before any classifier |
| Classifier | Autoencoder reconstruction loss | RNN / DSNN authentication |
| Text modes | N/A | Text-dependent + text-independent |
| Target population | General | Severely dysphonic speakers (DLS) |

---

## Common pitfalls

1. **Paired loading is always enforced.** `load_dataset()` always loads both audio and EEG for every trial, using the `eeg_index` column from the audio MAT to identify the correct EEG row. The `modality` field selects which signal is used during feature extraction — not which data is loaded. Subjects or trials missing either modality are silently dropped. This guarantees that voice-only, EEG-only, and fused runs operate on the **exact same set of subjects and trials**, making results directly comparable.

2. **Per-dimension [0,1] scaling before paraconsistent.** α and β are range-based and assume commensurable features in [0,1]. `score_feature_set()` min-max scales each dimension across all samples (audit M-1). Do **not** use per-sample sum-1 normalization — it lets large-magnitude descriptors (energy) dominate and pushes signed descriptors (Teager) outside [0,1], breaking the α domain.

3. **EEG and voice window sizes differ.** EEG at 1024 Hz needs different window parameters than voice at 44100 Hz. Do not reuse the same `window_size` across modalities.

4. **`jitter`/`shimmer` require voiced frames.** Unvoiced frames produce undefined period estimates. Filter by voicing flag before computing perturbation measures.

5. **Autoencoder path wires `snn-ae` and `ann-ae`.** `feature_extraction.autoencoder.model` accepts `snn-ae`, `ann-ae`, or `lstm-ae`; the thesis Phase 00 profiles use `snn-ae`/`ann-ae` (`lstm-ae` is the legacy Guayaquil extractor, unused by any profile).

6. **Text-independent split must not leak phrases.** Train and test splits must use disjoint phrase sets, not just disjoint utterances of the same phrase.

7. **Nested CV inner fold must not see test fold.** Hyperparameter selection must use inner-loop validation only.

8. **EER is NaN with grouped CV + degenerate fold.** `GenuineImpostorEERScorer` returns NaN when test speakers have no probes or only one speaker in fold. This is correct behaviour.

9. **`max_samples` truncation round-robins across subjects.** Samples are stored subject-contiguous (~130 trials/speaker), so a first-N truncation would keep only 2–3 speakers and break speaker-disjoint (GroupKFold) folds — especially nested CV, whose inner fold would then have fewer groups than splits (`GroupKFoldPolicy: number of unique groups is less than n_splits`). `load_dataset()` therefore selects the capped subset round-robin across subjects so every speaker is represented. Found by the smoke suite — see [Ground-Truth and Smoke Testing](../Guides/Ground-Truth-and-Smoke-Testing.md).

12. **A ranking table's sort direction is not self-evident from the data — verify it against the actual winner.** `e05_build_phase00_paraconsistent_tables.py` sorted `reverse=True` (descending by mean `d_truth`) for a long time before this was caught, on the reasoning that a ranked table "starts at the top". But `d_truth`/`d_penalized` are *distances* to the paraconsistent Truth vertex — lower is better — so descending order silently put each signal's actual winner **last**: the real EEG winner sat at rank 46 of 46. It also ranked on `d_truth`, not `d_penalized`, the metric `01_e05_phase00_rank.py` actually selects on — the two orders genuinely diverge (see [Core/Paraconsistent.md](../Core/Paraconsistent.md#selection-metric-contradiction-penalized-truth-distance)). The fix: sort ascending by `d_penalized`, emit both `d_penalized` and `d_truth` columns, and mark the true winner (cross-checked against `winners.json`) with a dagger rather than trusting rank 1. When any script writes a "ranked" table, check its sort direction and key against an independent source of truth — do not assume rank 1 is the winner just because the column is populated.

## Testing

Beyond the unit tests (`e05_*_gtest`), two extra layers guard this experiment:

- **Per-profile smoke runs** — `profiles/smoke/` mirrors all 315 profiles with tiny run parameters; `scripts/testing/run_e05_smoke.sh` runs each end-to-end to catch runtime errors compilation cannot. The mirror auto-regenerates via the CMake `e05_smoke_profiles` target when any source profile changes.
- **PyTorch / snnTorch parity** — layer-level numerical ground truth (Linear, activations, MSE/CE losses, LSTM, LifBPTT, Conv1d/2d, MaxPool).

Both are documented in [Ground-Truth and Smoke Testing](../Guides/Ground-Truth-and-Smoke-Testing.md).

9. **SQLite float32 blobs.** The 10.1117 database stores audio/EEG blobs as `float32`. Loaders detect encoding by byte-size checks; mismatches throw explicit runtime errors.

10. **`nn::Tensor` default is a 0-dim scalar** (`size()=1, rows()=0, cols()=0`). Guard audio/EEG samples with `rows() > 0 && cols() > 0`, not `size() > 0`.

11. **`discoverSubjects` regex must have a capturing group.** Use `"^S(\\d+)$"` not `".*"`.

---

## See also

- [Paraconsistent Feature Engineering](../Core/Paraconsistent.md) — α/β/D_truth formulas and API
- [Paraconsistent — Plain](../Core/Plain/Paraconsistent.md) — accessible explanation
- [LFCC](../Concepts/LFCC.md) — frequency scale used in handcrafted features
- [Imagined Speech and EEG](../Concepts/Imagined-Speech-and-EEG.md) — neuroscience background
- [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md) — SNN-AE / DSNN theory
- [LSTM and BPTT](../Concepts/LSTM-and-BPTT.md) — LSTM-AE theory
- [K-Fold Cross-Validation](../Concepts/K-Fold-Cross-Validation.md) — nested CV
- [Data Loaders](../Core/DataLoaders.md) — 10.1117 loader API
- [Research Context](../Research-Context.md) — thesis goals and full pipeline
- [Experiment04](./Experiment04.md) — prior congress paper experiment
- [Re-run Runbook](../Guides/Re-run-Runbook.md) — commands to regenerate every result
- [Engineering Fixes Log](../Guides/Engineering-Fixes-Log.md) — the D1-D6 decision log behind the current `d_penalized` metric, 208-profile grid, and re-run

---

## References

[A] R. C. Guido, "Paraconsistent feature engineering," *Knowledge-Based Systems*, 2018.

[B] S. Zhao et al., "EEG-based imagined speech recognition using deep learning," *IEEE Trans. Neural Syst. Rehabil. Eng.*, 2021.

[C] Dataset 10.1117/12.2255697: G. A. Pressel Coretto, I. E. Gareis, and H. L. Rufiner, "Open access database of EEG signals recorded during imagined speech," in *Proc. SPIE 10160, 12th Int. Symp. Medical Information Processing and Analysis*, 2017. [Online]. Available: https://doi.org/10.1117/12.2255697

[D] E. O. Neftci, H. Mostafa, and F. Zenke, "Surrogate gradient learning in spiking neural networks," *IEEE Signal Process. Mag.*, vol. 36, no. 6, pp. 51–63, 2019.

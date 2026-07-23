# Prompt — Multi-Objective Genetic Algorithm for Evolving Autoencoders (ANN and SNN)

## 1. Role and objective

You are an engineering agent working **inside an existing, mature codebase** (`software/nn/`, the thesis's C++20 SNN/ML framework). Your task is to implement a **multi-objective genetic algorithm (NSGA-II)** that evolves autoencoder architectures across two parallel, independent populations — one of classical artificial neural networks (ANN) and one of spiking neural networks (SNN) — selecting the individuals that produce the **best feature vectors according to Paraconsistent Feature Engineering**, subject to a **hard inference-latency constraint**.

The end goal of the thesis system is **biometric authentication of severely dysphonic speakers, enhanced by imagined-speech EEG** (see `software/nn/.wiki/Research-Context.md`): degraded/absent voice biometrics are compensated with EEG captured during imagined speech. Concretely this means **three signal tracks — `eeg`, `voice`, and `fused` (early/late)** — matching the existing `thesis` experiment's phase00/phase01 profiles (`software/nn/src/experiments/thesis/profiles/`, `software/nn/results/thesis/phase01/*.json`). This GA task targets the `eeg` and `voice`/`fused` autoencoder populations, not a voice-only "speaker biometrics" system — do not drop the EEG/imagined-speech framing when designing the genome or evaluation split.

**Raspberry Pi / embedded deployment is not an existing project target.** It does not appear anywhere in `software/nn/.wiki/`, `documentation/`, or the thesis code — treat the 1 s end-to-end latency ceiling and the target-hardware requirement in §4 as a **new constraint being introduced by this task**, not a previously established fact. If the actual target hardware is undecided or different, confirm with the user before hardcoding "Raspberry Pi" into configs, filenames, or calibration utilities.

---

## 2. PHASE 0 — Mandatory discovery (perform before writing any code)

> **Hard rule: do NOT reimplement anything that already exists in the codebase.**
> The paraconsistent analysis, neuron models, layers, training utilities, and data structures **are already implemented**. Your job is to *assemble and orchestrate*, not to recreate.

Signatures, module names, and API contracts are **not provided in this prompt** — you must discover them yourself, in this order:

1. **Project Wiki** (`software/nn/.wiki/`). Read `Core/Paraconsistent.md`, `Core/Plain/Paraconsistent.md`, `Research-Context.md`, `Experiments/Thesis.md`, and `Experiments/ParaconsistentBaseline.md` first. It is the source of truth for the design intent of each component.
2. **Already-implemented neural networks and experiment structure.** Do not start from a blank slate — the following are known to already exist and must be reused, not recreated:
   - Paraconsistent core primitives: `software/nn/include/paraconsistent/paraconsistent.hpp` + `software/nn/src/core/paraconsistent/paraconsistent.cpp` (`calculate_alpha`, `calculate_beta`, `calculate_certainty_degree_g1`, `calculate_contradiction_degree_g2`).
   - Thesis-specific `D_penalized` ranking layer: `software/nn/src/experiments/thesis/lib/include/ThesisParaconsistent.hpp`/`.cpp`, which already defines `ParaconsistentScore{label, alpha, beta, g1, g2, d_truth, d_penalized}` and the exact `kContradictionPenalty = 2 - sqrt(2)` constant used in §3.1.
   - ANN and SNN autoencoders: `software/nn/src/experiments/autoencoderRunner/` (build target `autoencoderRunner`, formerly "Experiment03") and `software/nn/src/core/models/autoencoder/` — `AudioWindowAutoencoder`, `EegWindowAutoencoder`, `FusedWindowAutoencoder`, `ProtocolAutoencoder` (ANN) and their spiking counterparts `AudioWindowSpikingAutoencoder`, `EegWindowSpikingAutoencoder`, `FusedWindowSpikingAutoencoder`, `ProtocolSpikingAutoencoder` (SNN), built on the generic `BaseAutoencoder.hpp`/`SpikingAutoencoder.hpp`.
   - SNN neuron models and surrogate-gradient training: `software/nn/include/layers/spiking/{Lif,LifBPTT,ISurrogateGradient,SurrogateGradient}.hpp`.
   - The `thesis` experiment's existing `phase00`/`phase01` structure (`software/nn/src/experiments/thesis/`, see its README) — this GA task should slot in as a new phase (e.g. `phase02`) or sibling experiment following the same profile/JSON-config convention (`src/experiments/thesis/profiles/phase00/`), not a standalone structure.

   Confirm the exact current signatures of these (they may have shifted since this prompt was written) and read them as a *pattern reference* for how to compose layers, instantiate neurons, run training, and collect metrics in this project.
3. **Source-code comments.** The library comments are substantive and often document preconditions, units, and side effects that do not appear in the signature.

**Known gaps — confirmed absent as of this writing, budget real implementation time for them:**
- No NSGA-II, genetic algorithm, or evolutionary-search code exists anywhere in the repo. This is genuinely new work.
- Spike-train reconstruction losses named in §5.2 (Van Rossum Distance, Victor–Purpura Distance, Spike Count Loss) do not exist. Only `software/nn/include/layers/losses/SpikeTimeLoss.hpp` exists as a related building block — extend it or add new loss modules, following its conventions.
- No dedicated hardware-latency-calibration utility exists yet, though the `thesis` phase00 profiles already model latency tiers (`profiles/phase00/p00_ae_snn_latency_{base,small,tiny}_{voice,eeg}.json`) — extend that pattern rather than inventing a new config shape.

**Phase 0 deliverable (before coding):** a short report listing, for every piece you intend to use — paraconsistent analysis, neurons, layers, training loop, data loading, timing instrumentation — the module, symbol name, signature, and the source where you confirmed it (Wiki, example code, or comment). For the three known gaps above, confirm they are still absent and propose where to create them, respecting the codebase conventions. **Do not invent APIs.** Where the Wiki and the code disagree, the code wins — but report the divergence.

---

## 3. Fitness

### 3.1 Primary objective — paraconsistent feature quality

Individual quality is measured by the penalized distance to the **Truth** vertex of the paraconsistent plane, as defined in the thesis (section 2.1.5.1). The chain, for reference — **use the existing implementation, do not reimplement it**:

- `α` = smallest intra-class similarity (vectors normalized to `[0,1]`)
- `β` = inter-class overlap ratio, `β = R / F`, with `F = N·(N−1)·X·T`
- `G₁ = α − β` (degree of certainty)
- `G₂ = α + β − 1` (degree of contradiction)
- `D₁,₀ = √((G₁ − 1)² + G₂²)` (distance to Truth)

**Selection metric:**

```
D_penalized = D₁,₀ + λ·|G₂|,     λ = 2 − √2 ≈ 0.5858
```

**Minimize.** `D_penalized = 0` only at Truth; the three degenerate vertices (Falsity, Ambiguity, Indefinition) all evaluate to exactly 2.

> **Why the penalty term is mandatory, not optional:** a poorly trained autoencoder producing near-constant output reaches `α = 1` and `β = 1` — the Ambiguity vertex — and scores `D₁,₀ = √2 ≈ 1.4142`, low enough to top the ranking while carrying no class information whatsoever. This **actually occurred** during the thesis experiments. Under GA selection pressure this degeneracy is not a rare accident: it is a local optimum the search will actively find. `D_penalized` is the defense against it and **must not be replaced by plain `D₁,₀` under any circumstances**.

### 3.2 Secondary objective — inference cost

Inference cost measured **on the target hardware or on a calibrated proxy** (see §4). Minimize.

### 3.3 Reconstruction error — sanity check, not an objective

Reconstruction error is **not an optimization objective**. It serves exclusively as a **sanity check**: proof that the latent vector actually encodes the signal rather than being an artifact detached from the input. Implement it as a **binary filter**: individuals whose reconstruction is worse than a threshold `τ_rec` are marked infeasible regardless of their `D_penalized`.

This is a second line of defense against the degeneracy in §3.1 and is intentionally redundant with it.

### 3.4 NSGA-II structure

Multi-objective optimization by Pareto dominance, **with no weighted scalarization**:

| # | Objective | Direction |
|---|-----------|-----------|
| 1 | `D_penalized` | minimize |
| 2 | inference cost (latency) | minimize |

Use non-dominated sorting into fronts plus crowding distance for diversity, per standard NSGA-II. The final output is the **Pareto front**, not a single individual.

**Constraints (constrained dominance — a feasible individual always dominates an infeasible one):**
- end-to-end latency above the ceiling → infeasible
- reconstruction worse than `τ_rec` → infeasible

---

## 4. Latency constraint

> **This section introduces a new project requirement.** No target-hardware or end-to-end latency ceiling is documented anywhere in `software/nn/.wiki/` or `documentation/` as of this writing. Confirm the target hardware and ceiling value with the user before treating them as fixed; the values below (Raspberry Pi, 1 s) are proposed defaults, not confirmed facts.

**Proposed end-to-end ceiling: 1 second**, on the target hardware (default assumption: Raspberry Pi — confirm before use), measuring the **complete authentication chain** — acquisition/preprocessing (wavelet or wavelet-packet, normalization), autoencoder inference, feature extraction, and response to the user.

Important implementation consequence: the autoencoder **does not get the whole second**. Its budget is

```
autoencoder_budget = 1000 ms − fixed_pipeline_cost
```

where `fixed_pipeline_cost` is constant with respect to the genome (it does not depend on the evolved individual).

**Therefore:** the GA must read `fixed_pipeline_cost` from configuration rather than assume it. Implement a calibration utility that measures this fixed cost once, with the signal already fixed, and persists the result. Until that value is measured, use an explicit configuration value and **log that it is an uncalibrated estimate**.

Apply the constraint as a **cheap pre-training screen** wherever possible: if the genome's estimated inference cost (from operation/parameter counts) already exceeds the budget, discard the individual without training it.

---

## 5. Experimental protocol

### 5.1 Genome

**The GA genome must combine the same attributes already varied across the phase00 AE profiles** (`software/nn/src/experiments/thesis/profiles/phase00/p00_ae_{ann,snn}_*.json`, parsed into `thesis::ThesisConfig::AutoencoderConfig` — confirm current field names/values in Phase 0, they may have shifted). Do not invent a new, unrelated set of architecture axes: phase00 already defines the dimensions this thesis treats as meaningful for AE feature quality, and the GA should search that same space rather than a generic one, so that GA results remain comparable to the phase00 baseline.

Phase00's actual AE search axes, to carry into the genome:

- `model`: population-defining, not a gene — `ann-ae` for the ANN population, `snn-ae` for the SNN population (§1).
- `modality`: `eeg` | `voice` (and `fused`, if in scope per §5.4) — run as separate GA studies per §5.4, not as a gene within one run, mirroring how phase00 splits profiles by modality.
- **width/size tier** — phase00 sweeps `encoder_layer_spec`/`decoder_layer_spec` as coupled pairs (`tiny`: 16→8, `small`: 32→16, `base`: 64→32). Treat this as the genome's primary architecture gene, either as the ordinal tier or, preferably, as the two integer widths directly (hidden width, **latent space dimension** — the axis with the largest impact on both objectives).
- `encoding` (SNN population only): `direct` | `latency` | `poisson`, matching `AutoencoderConfig::encoding`.
- `time_steps` and `voltage_threshold` (SNN population only): in phase00 these are **coupled to `encoding`**, not independently varied (`direct` → `time_steps=1, threshold=1.0`; `latency`/`poisson` → `time_steps=16, threshold=0.2`). **Decision to record explicitly:** keep them coupled to `encoding` as phase00 does, or promote them to independent genes. If promoted, that is a declared expansion of the phase00 search space, not a rediscovery of it — say so in the Phase 0 report.
- training hyperparameters (learning rate, etc.) — phase00 holds these **fixed** (`epochs=100`, `learning_rate=0.001`, `samples_per_batch=32`, ...) across all AE profiles. Per the loss-function decision below, default to keeping these fixed too unless there is a specific reason to evolve them; if evolved, declare it as an axis beyond phase00's original scope.

Everything phase00 keeps constant across AE profiles (encoder/decoder depth fixed at 2 layers each, activations fixed at `leaky`→`identity`, `firing_rate_reg_lambda/min/max` fixed, `k_folds`/`nested_cv`/seeds fixed) should likewise stay fixed in the GA unless there's a documented reason to open it up — opening additional axes is a scope decision to make explicitly, not a default.

**Decision to record explicitly:** is the loss function part of the genome, or fixed per population? Recommendation — **fix it per population** in the first study, so that architecture effects are not confounded with loss effects. If it is evolved, treat it as a declared axis and report that analysis separately.

#### 5.1.1 Size of the search space (measured from the codebase)

Counted from the enforced value lists in the source, **excluding layer counts and layer widths**. These are the numbers the GA budget must be sized against.

**GA genome, as configured today** (`profiles/pga_*.json`, bounds `hidden ∈ {16,32,64,128}`, `latent ∈ {8,16,32,64}`, constraint `latent < hidden` → 10 valid width pairs):

| Population | Distinct genomes | × `n_seeds` | Max AE trainings |
|---|---|---|---|
| SNN (10 pairs × 3 encodings) | **30** | 3 | 90 |
| ANN (10 pairs, encoding forced `direct`) | **10** | 3 | 30 |
| Excluding widths entirely | 3 (SNN encodings) + 1 (ANN) = **4** | — | — |

> **Budget warning — act on this.** Both profiles run `population_size=16 × (1+12 generations) = 208` evaluations against only **30** (SNN) / **10** (ANN) reachable genomes. The evaluation cache makes the ~180 redundant evaluations free, but NSGA-II then degenerates into an **exhaustive sweep**: selection, crossover and mutation cannot find anything enumeration would miss. Either widen the bounds (more widths, `evolve_temporal: true`, learning rate as a gene) or cut `generations` to ~3–4. Do not report a "genetic search" result that is really a full enumeration.

**AE-only scope** (handcrafted extraction, wavelets, scales, categories, descriptors and `lstm-ae` all excluded; backend fixed to xtensor; only AE-compatible optimizers and losses):

| Level | Count |
|---|---|
| Per modality-slot: `ann-ae` (1, encoding forced `direct`) + `snn-ae` × 3 encodings | 4 |
| Modality slots: eeg + voice + fused(early, late) | 4 |
| **AE extractor configs** | **16** |
| × 4 optimizers (adam, sgd, lion, schedule-free-adamw — all AE-compatible) | 64 |
| × reconstruction loss (encoding-determined: 2 for `direct`, 1 each for `poisson`/`latency` + opt-out baselines) | **128** |
| + phase01 classifier stage (× 16) | 2 048 |

**128** is the AE-only configuration space. Compatibility was verified against the code, not assumed:

- **Optimizers — all 4 are compatible.** `run_protocol_ae` forwards `optimizer_type` straight into `TrainerConfig`, `OptimizerFactory` supports all four, and every optimizer honors `attach_with_scales()` for the SNN biophysical parameters (R, C, V_th). No AE-specific exclusion exists.
- **Losses — 4 are wired, but the encoding fixes the choice.** The AE reconstruction path can instantiate `MSELossImpl`, `MAELossImpl`, `SpikeCountLossImpl` and `SpikeTimeLossImpl`. `CrossEntropyLoss` is classification and is excluded. The encoding↔loss pairing is an **enforced invariant**, not a free axis (`.wiki/Concepts/Spike-Encoding.md`): `direct → mse|mae`, `poisson → spikecount`, `latency → spiketime`. `validate()` rejects a mismatched pair, rejects spike losses for `ann-ae`/`lstm-ae` (they emit continuous values, never spikes), and rejects them for `direct` (analog, no spikes). `mse`/`mae` stay selectable under `poisson`/`latency` as an explicit opt-out baseline.

> **Implementation note (resolved).** The Trainer's loss is a *compile-time* template parameter (`Trainer<ModelType, LossType>`, defaulting to `MSELossImpl`), and the old `AutoencoderConfig::loss_type` string was never read — so only MSE was ever trained. `ThesisConfig::AutoencoderConfig::ae_loss_type` (`mse | mae | spikecount | spiketime`, validated) now dispatches to a concrete `Trainer` instantiation in `ThesisFeatureExtraction`. Set it explicitly per run; **keep it fixed per population** (§5.1) rather than evolving it.
>
> `spiketime` carries a **layout requirement**: `SpikeTimeLossImpl` indexes rows as `t*B + b`, i.e. a time-major `(T*B, F)` tensor, whereas the default AE sample is a single `(1, D)` frame that the Trainer stacks into `(B, D)` — batch rows with no time axis. Feeding that would silently reinterpret unrelated samples as timesteps. The Trainer cannot build the right layout itself (`create_batch` makes a 3-D `(B, T, C)` tensor for multi-row samples, and it reshuffles indices every epoch). The AE trainer therefore **pre-interleaves the batch**: each training sample is a whole group of `samples_per_batch` inputs laid out as `(T*g, D)` with `row = t*g + b`, and the Trainer runs one group per step. **`batch_size` is fully honoured** — it lives in the sample layout rather than in `create_batch` — and a trailing partial group is fine because the loss derives `B = rows / T`.

**Framework-wide categorical axes** (the space the GA could in principle be extended into, i.e. *before* the AE-only exclusions above):

| Axis | Values | Count |
|---|---|---|
| Modality | voice, eeg, fused | 3 |
| Fusion mode (fused only) | early, late | 2 |
| Wavelet | haar + daub4…daub46 (even) | 23 |
| Scale | bark, mel, lfcc (**EEG: lfcc only**) | 3 |
| Category | cepstral true/false | 2 |
| Descriptors | energy, zcr, entropy, teager, jitter, shimmer | 6 (subset) |
| AE model | ann-ae, snn-ae, lstm-ae | 3 |
| Spike encoding (snn only) | direct, latency, poisson | 3 |
| Classifier | rnn, dsnn | 2 |
| Text mode | dependent, independent | 2 |
| Optimizer | adam, sgd, lion, schedule-free-adamw | 4 |
| Batch norm (dsnn only) | none, threshold-dependent | 2 |
| Standardize / nested CV | bool × bool | 4 |
| Surrogate gradient | Exponential, Boxcar | 2 |
| Losses | MSE, MAE, CrossEntropy, SpikeCount, SpikeTime | 5 |
| Initializer | xavier, kaiming_snn | 2 |

Resulting pipeline sizes:

- **Handcrafted extractors:** eeg 23×1×2 = 46 · voice 23×3×2 = 138 · fused 23×3×2×2 = 276 → **460**
- **Autoencoder extractors:** (ann + lstm + snn×3) = 5 per modality → eeg 5 + voice 5 + fused 10 = **20**
- **Classifier stage:** rnn/dsnn × dep/indep × standardize × nested_cv = **16** (×4 optimizers = 64; ×2 batch-norm, dsnn only → up to 128)
- **Full thesis pipeline:** (460 + 20) × 16 = **7 680**; ×4 optimizers = **30 720**
- **Guayaquil:** 1 lstm-ae + 3 architectures × 3 encodings = **10**
- **autoencoderRunner:** 4 dataset types × 2 families = **8** concrete AE classes

**Three caveats that keep these honest:**

1. **Not all products are reachable.** The code enforces real exclusions: EEG rejects bark/mel (cochlear scales, provably degenerate to lfcc there), `fusion_mode` applies only when modality is `fused`, spike `encoding` is ignored by `ann-ae`/`lstm-ae`, and batch-norm / firing-rate regularization are `dsnn`-only. The totals above already apply the modality/scale and encoding gates; treat the rest as an upper bound.
2. **Loss is not a free axis in the thesis pipeline.** `ae_cfg.loss_type = "mse"` is hard-coded in `ThesisFeatureExtraction.cpp`, not profile-driven — so the 5 losses are available in the framework but pinned in practice. This is consistent with §5.2's "fix loss per population" recommendation.
3. **Descriptors are the hidden multiplier.** Counted above as one fixed set. `descriptors` is a `vector<string>` with no validation beyond non-empty, so swept as subsets it is 2⁶−1 = **63**, which would take handcrafted from 460 to **28 980**. Every profile in the repo uses one fixed list — the axis is declared but never swept. Do not open it without an explicit scope decision.

The 460 handcrafted figure is a **superset** of the executed phase00 grid (276 handcrafted + 24 AE): fused handcrafted was never swept and AE sizes were fixed tiers.

### 5.2 Loss functions per population

**ANN population** — primary loss: **MSE**. Alternatives recorded as experimental variants: MAE, Huber.

**SNN population** — training via **surrogate gradient**, using the existing `software/nn/include/layers/spiking/SurrogateGradient.hpp`/`ISurrogateGradient.hpp` (confirm in Phase 0 which surrogate shapes it currently supports, e.g. SuperSpike or exponential surrogate, before assuming both are available). Spike-reconstruction evaluation metrics: Spike Count Loss, Van Rossum Distance, Victor–Purpura Distance — **none of these exist in the codebase yet**; only `software/nn/include/layers/losses/SpikeTimeLoss.hpp` exists as a related building block. Implement the missing ones as new loss modules following that file's conventions, and say so explicitly in the Phase 0 report rather than assuming they can be reused.

> **Mandatory hyperparameters, currently undefined:** Van Rossum requires the exponential filter time constant `τ`; Victor–Purpura requires the temporal cost `q`. Both metrics change behavior qualitatively with these values. **Fix them in configuration, document the chosen value and its rationale, and do not leave them implicit in the code.** If they are to be evolved, that must be a declared decision, not a side effect.
>
> Cost note: Victor–Purpura is computationally expensive. Consider using it only in the final evaluation of the Pareto front, not every generation.

**SNN-specific caution:** the thesis documents that an effective learning rate incompatible with the biophysical parameters of the spiking neurons was the root cause of the `α = β = 1` degeneracy. The GA will explore learning rates freely. Ensure that (a) the search range is compatible with the biophysical parameters in use, and (b) `α`, `β`, `G₁`, and `G₂` are **logged individually for every individual**, not just the final `D_penalized` — without this, degenerate cases are indistinguishable from ordinary poor cases in post-mortem analysis.

### 5.3 ANN/SNN comparability

The two populations do not produce directly comparable quantities: one operates on continuous signals, the other on spike trains. **Define and document the common evaluation space.** Recommendation: `D_penalized` is already naturally comparable, since it is computed over feature vectors normalized to `[0,1]` regardless of the technology that produced them — this is where the two populations meet. Reconstruction error (§3.3), by contrast, is **not** comparable across them; use separate `τ_rec` thresholds per population and never compare their absolute values across technologies.

Evolve the two populations **separately** (no crossover between them) and compare only the resulting Pareto fronts.

### 5.4 Signal tracks

Following the existing `thesis` phase00/phase01 convention (`software/nn/results/thesis/phase01/*.json` filenames), run and report the GA separately per signal track: `eeg`, `voice`, and `fused` (early/late fusion, if both are in scope — confirm with the existing `AutoencoderRunnerDatasetType` enum in Phase 0). Do not collapse these into a single undifferentiated "speaker audio" population — EEG is a first-class track in this thesis, not an auxiliary signal.

### 5.5 Training budget and reproducibility

- **Fixed budget per individual**: identical maximum epoch count for all, with a declared early-stopping criterion. Without this, "training cost" and "convergence time" are not fairly measurable.
- **Seeds**: each individual is trained with `n_seeds` distinct seeds (initial recommendation: 3). The individual's `D_penalized` is the **mean** across seeds; also record the standard deviation, which is the run-to-run stability measure.
- This multiplies GA cost by `n_seeds` — size the population and generation count accordingly.
- Every run must be reproducible from a versioned configuration file.

---

## 6. Deliverables

1. Phase 0 report (§2).
2. NSGA-II implementation integrated with the existing components, following the patterns observed in the already-implemented networks.
3. Fitness evaluation module: `D_penalized`, inference cost, reconstruction sanity filter.
4. Calibration utility for `fixed_pipeline_cost`.
5. Versioned configuration containing all hyperparameters, including `λ`, `τ` (Van Rossum), `q` (Victor–Purpura), `τ_rec`, `n_seeds`, epoch budget, latency ceiling, **optimizer**, and **`ae_loss_type` (`mse` | `mae`)** — the last two are the axes that make the AE space 128 rather than 16.
6. Per-individual logs containing at minimum: genome, `α`, `β`, `G₁`, `G₂`, `D₁,₀`, `D_penalized`, reconstruction error, measured latency, seed, feasibility.
7. Final Pareto fronts for both populations, exported in a queryable format.

---

## 7. Acceptance criteria

- No already-existing component in the codebase was reimplemented.
- `D_penalized` correctly reproduces the thesis reference cases: `(α,β) = (1,1)` → `= 2.0000` (the Ambiguity vertex; `λ = 2−√2` is chosen precisely so every non-Truth vertex scores exactly 2, consistent with §3.1 — an earlier draft's `2.4142` was wrong); `(α,β) = (0.92, 0.075)` → `≈ 0.1580`.
- An individual with constant output is ranked **worst** in the population, not best.
- No individual on the final Pareto front violates the end-to-end latency ceiling.
- Re-running from the same configuration and seeds reproduces identical results.
- **The GA budget is smaller than the reachable search space** (§5.1.1). If `population_size × (1 + generations)` exceeds the number of distinct genomes the bounds allow, the run is an exhaustive enumeration wearing a GA's clothes — either widen the bounds or shrink the budget before reporting it as an evolutionary search.


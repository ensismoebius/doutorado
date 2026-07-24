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
- Spike-reconstruction losses: `SpikeCountLoss` and `SpikeTimeLoss` **now exist and are wired** into the AE path (rate→`spikecount`, latency→`spiketime`, enforced; see §5.1.1). `Van Rossum Distance` and `Victor–Purpura Distance` (§5.2) still do **not** exist — add them as new loss modules following `SpikeTimeLoss.hpp` conventions if the Pareto-front evaluation needs them.
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
- **architecture (free depth + free per-layer width)** — this is the primary architecture gene and it is **fully open**: there are **no pre-defined layer tiers**. The genome carries an arbitrary-length list of encoder widths, `encoder_widths`; both the number of layers AND each layer's neuron count come from the DNA. The last element is the latent (bottleneck); the decoder is the mirror image. Example genomes across generations: `{3,2,1}` (256→3→2→1, decoder 1→2→3→256) then `{10,5,4,2}` (256→10→5→4→2, decoder mirrored). The **one** structural invariant is that widths strictly decrease (each layer compresses toward the bottleneck — the defining property of an autoencoder); `repair_widths` enforces it. This supersedes phase00's fixed `tiny/small/base` tiers — those become just three points the free search can reach, not the search space.
- `encoding` (SNN population only): `direct` | `latency` | `poisson`, matching `AutoencoderConfig::encoding`.
- `time_steps` and `voltage_threshold` (SNN population only): in phase00 these are **coupled to `encoding`**, not independently varied (`direct` → `time_steps=1, threshold=1.0`; `latency`/`poisson` → `time_steps=16, threshold=0.2`). **Decision to record explicitly:** keep them coupled to `encoding` as phase00 does, or promote them to independent genes. If promoted, that is a declared expansion of the phase00 search space, not a rediscovery of it — say so in the Phase 0 report.
- training hyperparameters (learning rate, etc.) — phase00 holds these **fixed** (`epochs=100`, `learning_rate=0.001`, `samples_per_batch=32`, ...) across all AE profiles. Per the loss-function decision below, default to keeping these fixed too unless there is a specific reason to evolve them; if evolved, declare it as an axis beyond phase00's original scope.

Everything phase00 keeps constant across AE profiles (encoder/decoder depth fixed at 2 layers each, activations fixed at `leaky`→`identity`, `firing_rate_reg_lambda/min/max` fixed, `k_folds`/`nested_cv`/seeds fixed) should likewise stay fixed in the GA unless there's a documented reason to open it up — opening additional axes is a scope decision to make explicitly, not a default.

**Decision to record explicitly:** is the loss function part of the genome, or fixed per population? Recommendation — **fix it per population** in the first study, so that architecture effects are not confounded with loss effects. If it is evolved, treat it as a declared axis and report that analysis separately.

#### 5.1.1 Size of the search space (measured from the codebase)

Counted from the enforced value lists in the source, **excluding layer counts and layer widths**. These are the numbers the GA budget must be sized against.

**GA genome architecture space — free depth + free per-layer width.** The genome is an
arbitrary-length list of strictly-decreasing encoder widths (§5.1). With bounds
`min_layers..max_layers` layers and widths drawn from `[min_width, max_width]`, the number
of distinct architectures is the count of strictly-decreasing integer sequences:

```
architectures = Σ_{d=min_layers}^{max_layers}  C(width_room, d),   width_room = max_width − min_width + 1
```

For the shipped bounds (`min_layers=1, max_layers=6, min_width=1, max_width=128` → `width_room=128`):

| d (layers) | C(128, d) |
|---|---|
| 1 | 128 |
| 2 | 8 128 |
| 3 | 341 376 |
| 4 | ≈ 1.06 × 10⁷ |
| 5 | ≈ 2.64 × 10⁸ |
| 6 | ≈ 5.42 × 10⁹ |
| **Σ (1–6)** | **≈ 5.7 × 10⁹ architectures** |

× SNN encodings (×3) and × `n_seeds` on top of that. **The architecture space is now
effectively unbounded relative to any feasible GA budget** — the opposite of the
fixed-tier design it replaces.

> **This inverts the earlier budget warning.** Previously the genome had only 30 (SNN) / 10
> (ANN) reachable shapes, so `population_size × (1 + generations) = 208` evaluations made
> NSGA-II a disguised exhaustive sweep. With a free architecture the space is ~10⁹, so the
> GA is now a **genuine search** — crossover and mutation matter, and the evaluation cache
> saves real recomputation rather than papering over enumeration. Size `population_size`
> and `generations` for *coverage of a huge space*, not to avoid re-enumeration. The
> §7 acceptance criterion on budget-vs-space is now satisfied by construction.

**AE-only scope** (handcrafted extraction, wavelets, scales, categories, descriptors and `lstm-ae` all excluded; backend fixed to xtensor; only AE-compatible optimizers and losses). Because the encoding↔loss pairing is now an **enforced invariant** (`validate()` throws on a mismatch), `loss` is not a free ×2 axis — it is bound to the `(model, encoding)` pair. The unit of counting is therefore the **valid `(model, encoding, loss)` triple**:

| model | encoding | legal `ae_loss_type` | triples |
|---|---|---|---|
| `ann-ae` | `direct` (forced) | `mse`, `mae` | 2 |
| `snn-ae` | `direct` | `mse`, `mae` | 2 |
| `snn-ae` | `latency` | `mse`, `mae`, `spiketime` | 3 |
| `snn-ae` | `poisson` | `mse`, `mae`, `spikecount` | 3 |
| | | **total** | **10** |

| Level | Count |
|---|---|
| Valid `(model, encoding, loss)` triples per modality-slot | **10** |
| × 4 modality slots (eeg, voice, fused-early, fused-late) | 40 |
| × 4 optimizers (adam, sgd, lion, schedule-free-adamw — all AE-compatible) | **160** |
| × architecture (free depth+width, ~5.7×10⁹ shapes — §5.1.1) | **effectively unbounded** |
| (+ phase01 classifier stage, × 16) | 25 600 |

**160** is the AE-only configuration space at the granularity the earlier draft called "128" (model × encoding × modality × optimizer × loss, widths excluded). The rise from 128 → 160 is the two spike-loss combos (`latency+spiketime`, `poisson+spikecount`) that are now **wired and reachable**, × 4 slots × 4 optimizers = +32. Everything was verified against the code, not assumed:

- **Optimizers — all 4 are compatible.** `run_protocol_ae` forwards `optimizer_type` straight into `TrainerConfig`, `OptimizerFactory` supports all four, and every optimizer honors `attach_with_scales()` for the SNN biophysical parameters (R, C, V_th). No AE-specific exclusion exists.
- **Losses — 4 are wired, and the encoding fixes the choice.** The AE reconstruction path can instantiate `MSELossImpl`, `MAELossImpl`, `SpikeCountLossImpl` and `SpikeTimeLossImpl`. `CrossEntropyLoss` is classification and is excluded. The invariant (`.wiki/Concepts/Spike-Encoding.md`): `direct → mse|mae`, `poisson → spikecount`, `latency → spiketime`. `validate()` **rejects** a mismatched pair (`poisson+spiketime`, `latency+spikecount`), rejects spike losses for `ann-ae`/`lstm-ae` (they emit continuous values, never spikes), rejects them for `direct` (analog, no spikes), and rejects an empty loss (no silent fallback to `mse`). `mse`/`mae` stay selectable under `poisson`/`latency` as an explicit opt-out baseline — the two rows with 3 legal losses.
- **GA consequence:** because a run fixes one `ae_loss_type`, and mutation may flip the SNN genome's `encoding`, a spike-loss run must pin `encoding_choices` to the single compatible encoding (`spiketime`→`latency`, `spikecount`→`poisson`) or `validate()` will throw mid-run. The shipped `pga_snn_eeg_spike{count,time}.json` profiles do exactly this.

> **Implementation note (resolved).** The Trainer's loss is a *compile-time* template parameter (`Trainer<ModelType, LossType>`, defaulting to `MSELossImpl`), and the old `AutoencoderConfig::loss_type` string was never read — so only MSE was ever trained. `ThesisConfig::AutoencoderConfig::ae_loss_type` (`mse | mae | spikecount | spiketime`, validated) now dispatches to a concrete `Trainer` instantiation in `ThesisFeatureExtraction`. Set it explicitly per run; **keep it fixed per population** (§5.1) rather than evolving it.
>
> **Silent-failure guard (`spikecount` / `spiketime`).** A spike loss can produce an
> all-zero gradient and still run to completion: `SpikeTimeLoss` attaches its gradient
> only at the predicted first-spike row, so a unit that never crosses threshold receives
> nothing. If that holds for every batch the autoencoder trains on nothing and emits
> features from an untrained model. This is **configuration-dependent** — measured on a
> synthetic set, `lr=0.01` was live across 20 seeds and every batch size while `lr=0.001`
> deadlocked at the same threshold, and encoder firing-rate regularization did *not*
> rescue it. The AE path therefore wraps spike losses in a liveness guard and **throws**
> when every batch was zero, naming cause and remedy. Treat a thrown run as a
> configuration bug to fix (raise lr, lower `voltage_threshold`, raise `time_steps`), not
> as a failed individual to score.
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
2. **Loss is profile-driven but encoding-bound.** `ThesisConfig::AutoencoderConfig::ae_loss_type` (`mse|mae|spikecount|spiketime`) now selects the AE reconstruction loss — the earlier hard-coded `ae_cfg.loss_type = "mse"` is gone. It is not a *free* ×5 axis, though: `CrossEntropy` is excluded, and the encoding↔loss invariant collapses the choice to the 10 valid triples above. Still consistent with §5.2's "fix loss per population" recommendation — pin one per run.
3. **Descriptors are the hidden multiplier.** Counted above as one fixed set. `descriptors` is a `vector<string>` with no validation beyond non-empty, so swept as subsets it is 2⁶−1 = **63**, which would take handcrafted from 460 to **28 980**. Every profile in the repo uses one fixed list — the axis is declared but never swept. Do not open it without an explicit scope decision.

The 460 handcrafted figure is a **superset** of the executed phase00 grid (276 handcrafted + 24 AE): fused handcrafted was never swept and AE sizes were fixed tiers.

### 5.2 Loss functions per population

**ANN population** — primary loss: **MSE**. Alternatives recorded as experimental variants: MAE, Huber.

**SNN population** — training via **surrogate gradient** (Exponential or Boxcar; confirm in Phase 0 which shapes are available). Spike-reconstruction losses: **`SpikeCountLoss` (rate/`poisson`) and `SpikeTimeLoss` (latency) now exist and are wired** into the AE reconstruction path, bound to the encoding by the invariant in §5.1.1. `Van Rossum Distance` and `Victor–Purpura Distance` do **not** exist yet — implement them as new loss modules following `SpikeTimeLoss.hpp` conventions if the final Pareto-front evaluation needs them, and say so in the Phase 0 report.

> **`SpikeTimeLoss` no-spike deadlock (must handle).** Its gradient is written only at the predicted first-spike row, so a unit that never crosses threshold gets **no gradient at all**. If that holds for every batch, the AE trains on nothing and emits features from an untrained model — silently. The wired path guards this with a liveness check that **throws** when every batch's gradient was all-zero (see §5.1.1). Do not remove that guard; treat a thrown run as a config to fix (raise lr, lower `voltage_threshold`, raise `time_steps`), not a failed individual to score.

> **Mandatory hyperparameters, currently undefined:** Van Rossum requires the exponential filter time constant `τ`; Victor–Purpura requires the temporal cost `q`. Both metrics change behavior qualitatively with these values. **Fix them in configuration, document the chosen value and its rationale, and do not leave them implicit in the code.** If they are to be evolved, that must be a declared decision, not a side effect.
>
> Cost note: Victor–Purpura is computationally expensive. Consider using it only in the final evaluation of the Pareto front, not every generation.

**SNN-specific caution:** the thesis documents that an effective learning rate incompatible with the biophysical parameters of the spiking neurons was the root cause of the `α = β = 1` degeneracy. The GA will explore learning rates freely. Ensure that (a) the search range is compatible with the biophysical parameters in use, and (b) `α`, `β`, `G₁`, and `G₂` are **logged individually for every individual**, not just the final `D_penalized` — without this, degenerate cases are indistinguishable from ordinary poor cases in post-mortem analysis.

### 5.3 ANN/SNN comparability

The two populations do not produce directly comparable quantities: one operates on continuous signals, the other on spike trains. **Define and document the common evaluation space.** Recommendation: `D_penalized` is already naturally comparable, since it is computed over feature vectors normalized to `[0,1]` regardless of the technology that produced them — this is where the two populations meet. Reconstruction error (§3.3), by contrast, is **not** comparable across them; use separate `τ_rec` thresholds per population and never compare their absolute values across technologies.

Evolve the two populations **separately** (no crossover between them) and compare only the resulting Pareto fronts.

### 5.4 Signal tracks

Following the existing `thesis` phase00/phase01 convention (`software/nn/results/thesis/phase01/*.json` filenames), run and report the GA separately per signal track: `eeg`, `voice`, and `fused` (early/late fusion, if both are in scope — confirm with the existing `AutoencoderRunnerDatasetType` enum in Phase 0). Do not collapse these into a single undifferentiated "speaker audio" population — EEG is a first-class track in this thesis, not an auxiliary signal.

### 5.4.1 Compute budget — sample cap (`dataset.max_samples`)

A full run of one population is `population_size × (1 + generations) × n_seeds ≈ 16 × 13 × 3 = 624` autoencoder trainings, and cost is **linear in the sample count** (measured). At the full 1974 samples the whole 12-profile sweep is ≈ **86 h**; that is why the shipped profiles cap `dataset.max_samples`.

**Measured per-training cost** (100 epochs, this hardware, `max-performance` CPU preset): SNN ≈ 5.6 s/training at 200 samples (averaged over a real 16-genome population, tail included); ANN ≈ 2.2 s. Linear scaling gives, for the whole sweep (5 ANN + 7 SNN profiles):

| `max_samples` | samples/subject (15 subj) | est. total, all 12 (upper bound) |
|---|---|---|
| 200 | 13 | ~8.7 h |
| 300 | 20 | ~13 h |
| 400 | 26 | ~17 h |
| 500 | 33 | ~22 h |
| **550 (shipped)** | **36** | **~24 h** |
| 1974 (full) | 131 | ~86 h |

The estimate is an **upper bound**: objective 2 (inference cost) applies selection pressure toward cheaper genomes, so later generations run faster than the random initial population these numbers are measured on.

Two facts that make this safe:
- **`max_samples` is stratified.** `apply_max_samples` (ThesisDataset.cpp) is round-robin across subjects, so `max_samples=400` keeps ~26 balanced samples from *every* subject — no class is dropped, and the paraconsistent α/β (which need all classes) stay well-formed. Do not go below ~15/subject (~225 total) or the metric gets noisy.
- **Cost is dominated by the free-architecture SNN tail** (6-layer × wide × `time_steps=16` genomes), not by any single knob. If a run overruns, the cheap additional levers are `n_seeds` (3→1 is a free 3×, losing only the stability std) and `generations`, before cutting samples further.

Shipped default: **`max_samples=550`** on all 12 profiles → ~24 h nominal for the full sweep (~36 balanced samples/subject — the most data that fits the window). Because the estimate is an upper bound, the real wall-clock should land at or under 24 h. Lower toward 300–400 for faster iteration; the paraconsistent floor is ~15/subject (~225 total).

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
5. Versioned configuration containing all hyperparameters, including `λ`, `τ` (Van Rossum), `q` (Victor–Purpura), `τ_rec`, `n_seeds`, epoch budget, latency ceiling, **optimizer**, and **`ae_loss_type` (`mse` | `mae` | `spikecount` | `spiketime`, encoding-bound per §5.1.1)** — optimizer and loss are the axes that expand the AE space from 16 (model×encoding×modality) to 160.
6. Per-individual logs containing at minimum: genome, `α`, `β`, `G₁`, `G₂`, `D₁,₀`, `D_penalized`, reconstruction error, measured latency, seed, feasibility.
7. Final Pareto fronts for both populations, exported in a queryable format.

---

## 7. Acceptance criteria

- No already-existing component in the codebase was reimplemented.
- `D_penalized` correctly reproduces the thesis reference cases: `(α,β) = (1,1)` → `= 2.0000` (the Ambiguity vertex; `λ = 2−√2` is chosen precisely so every non-Truth vertex scores exactly 2, consistent with §3.1 — an earlier draft's `2.4142` was wrong); `(α,β) = (0.92, 0.075)` → `≈ 0.1580`.
- An individual with constant output is ranked **worst** in the population, not best.
- No individual on the final Pareto front violates the end-to-end latency ceiling.
- Re-running from the same configuration and seeds reproduces identical results.
- **The GA budget is smaller than the reachable search space** (§5.1.1) — satisfied by construction now that architecture is free (depth + per-layer width), giving ~5.7×10⁹ shapes against a budget of a few hundred evaluations. There are **no pre-defined layer configurations**: layer count and per-layer neuron count both come from the DNA, and a genome like `{3,2,1}` in one generation and `{10,5,4,2}` in the next are both reachable and legal.


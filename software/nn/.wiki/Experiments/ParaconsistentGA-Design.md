# paraconsistentGA — Design Specification

The full design rationale, fitness definition, search-space sizing, and compute budget for
the [paraconsistentGA](ParaconsistentGA.md) experiment. This page was the standalone design
prompt (`ga.md`) before it was folded into the wiki; the **§ numbering is preserved** because
source comments across `src/experiments/paraconsistentGA/` and the thesis AE path cite it as
`ParaconsistentGA-Design.md §N`. For the didactic overview and data-flow read
[paraconsistentGA](ParaconsistentGA.md) first; this is the reference detail behind it.

---

## 1. Purpose

paraconsistentGA is a **multi-objective genetic algorithm (NSGA-II)** that evolves
autoencoder architectures across two parallel, independent populations — one of classical
artificial neural networks (ANN) and one of spiking neural networks (SNN) — selecting the
individuals that produce the **best feature vectors according to Paraconsistent Feature
Engineering** ([Paraconsistent Logic](../Core/Paraconsistent.md)), subject to an
inference-latency constraint.

It serves the thesis system: **biometric authentication of severely dysphonic speakers,
enhanced by imagined-speech EEG** ([Research Context](../Research-Context.md)). Degraded or
absent voice biometrics are compensated with EEG captured during imagined speech, so there
are **three signal tracks — `eeg`, `voice`, and `fused` (early/late)** — matching the
[Thesis](Thesis.md) experiment's phase00/phase01 profiles. The GA is run separately per
track (§5.4); EEG is a first-class track, not an auxiliary signal.

> **Raspberry Pi / embedded deployment is not a confirmed project target.** It appears
> nowhere in the wiki, `documentation/`, or the thesis code. The 1 s end-to-end latency
> ceiling and target hardware in §4 are a **design assumption**, not an established fact —
> the latency proxy is uncalibrated and every Pareto JSON says so.

---

## 2. Reuse map (what it builds on)

**Hard rule honoured: nothing already in the codebase was reimplemented** — the experiment
assembles and orchestrates existing paraconsistent analysis, neuron models, layers, training
utilities, and data structures. The reuse surface (verified against the code; see also
`src/experiments/paraconsistentGA/PHASE0.md`):

| Need | Symbol | Source |
|------|--------|--------|
| Paraconsistent α/β/G1/G2 | `calculate_alpha/beta/certainty_degree_g1/contradiction_degree_g2` | `include/paraconsistent/paraconsistent.hpp` |
| `d_penalized` ranking | `thesis::score_feature_set`, `ParaconsistentScore`, `kContradictionPenalty = 2−√2` | `thesis/lib/include/ThesisParaconsistent.hpp` |
| AE train + latent extraction | `thesis::extract_features` (Protocol ANN-AE / SNN-AE) | `thesis/lib/include/ThesisFeatureExtraction.hpp` |
| Config schema | `thesis::ThesisConfig::from_json` incl. `AutoencoderConfig` | `thesis/lib/include/ThesisConfig.hpp` |
| Dataset load | `thesis::load_dataset` → `ThesisDatasetView` | `thesis/lib/include/ThesisDataset.hpp` |
| SNN neurons / surrogate | `LifBPTT`, `SurrogateGradient` | `include/layers/spiking/` |

**Genuinely new work:** NSGA-II (`GaNsga2.{hpp,cpp}`) — no evolutionary-search code existed
in the repo before. Spike-reconstruction losses `SpikeCountLoss`/`SpikeTimeLoss` **now exist
and are wired** into the AE path (§5.1.1). `Van Rossum` and `Victor–Purpura` distances do
**not** exist — add them as new loss modules following `SpikeTimeLoss.hpp` if the final
Pareto-front evaluation needs them (§5.2). No hardware-latency calibration utility exists yet.

---

## 3. Fitness

### 3.1 Primary objective — paraconsistent feature quality

Individual quality is the penalized distance to the **Truth** vertex of the paraconsistent
plane (thesis §2.1.5.1), computed by the reused implementation:

- `α` = smallest intra-class similarity (vectors normalized to `[0,1]`)
- `β` = inter-class overlap ratio, `β = R / F`, with `F = N·(N−1)·X·T`
- `G₁ = α − β` (degree of certainty); `G₂ = α + β − 1` (degree of contradiction)
- `D₁,₀ = √((G₁ − 1)² + G₂²)` (distance to Truth)

**Selection metric (minimize):** `D_penalized = D₁,₀ + λ·|G₂|`, `λ = 2 − √2 ≈ 0.5858`.
`D_penalized = 0` only at Truth; the three degenerate vertices (Falsity, Ambiguity,
Indefinition) all evaluate to exactly **2**.

> **Why the penalty is mandatory:** a poorly trained autoencoder producing near-constant
> output reaches `α = β = 1` — the Ambiguity vertex — and scores `D₁,₀ = √2 ≈ 1.4142`, low
> enough to top the ranking while carrying no class information. This **actually occurred**
> in the thesis experiments; under GA selection pressure it is a local optimum the search
> actively finds. `D_penalized` is the defense and must not be replaced by plain `D₁,₀`.

### 3.2 Secondary objective — inference cost

A structural proxy (encoder MACs × `time_steps`), minimized. See §4 for why it is a proxy.

### 3.3 Reconstruction error — sanity check, not an objective

Reconstruction quality is a **binary feasibility filter**, not an objective: proof the latent
encodes the signal rather than being detached from it. Because the reused `extract_features`
returns latents (not decoder loss), this is realised as a **latent-collapse guard** — mean
per-dimension latent std must exceed `τ_rec` — which directly detects the α=β=1 degeneracy of
§3.1. It is a second, intentionally-redundant line of defense.

### 3.4 NSGA-II structure

Multi-objective optimization by Pareto dominance, **no weighted scalarization**:

| # | Objective | Direction |
|---|-----------|-----------|
| 1 | `D_penalized` | minimize |
| 2 | inference cost | minimize |

Non-dominated sorting into fronts + crowding distance; output is the **Pareto front**, not a
single individual. Constraints via **constrained dominance** (a feasible individual always
dominates an infeasible one): over-budget latency → infeasible; latent-collapse (recon worse
than `τ_rec`) → infeasible. See [Multi-Objective Optimisation](../Concepts/Multi-Objective-Optimisation.md).

---

## 4. Latency constraint

> **Design assumption, not a confirmed requirement.** No target-hardware or latency ceiling
> is documented for the thesis. The values below (Raspberry Pi, 1 s) are proposed defaults.

**Proposed end-to-end ceiling: 1 second**, on the target hardware, over the complete
authentication chain — acquisition/preprocessing (wavelet, normalization), autoencoder
inference, feature extraction, response. The autoencoder does not get the whole second:

```
autoencoder_budget = 1000 ms − fixed_pipeline_cost
```

where `fixed_pipeline_cost` is genome-independent. The GA reads it from configuration; until
a calibration utility measures it on real hardware, the value is an explicit config field and
every run **logs that it is uncalibrated**. The constraint is applied as a **cheap
pre-training screen**: a genome whose estimated inference cost already exceeds the budget is
discarded without training (`exceeds_latency_budget`). The proxy→ms factor (`ns_per_mac`) is
uncalibrated; treat `est_latency_ms` as relative, not real milliseconds.

---

## 5. Experimental protocol

### 5.1 Genome

- `model`: population-defining, not a gene — `ann-ae` / `snn-ae`.
- `modality`: `eeg | voice | fused` — separate runs per track (§5.4), not a gene.
- **architecture (free depth + free per-layer width)** — the primary architecture gene,
  fully open with **no pre-defined layer tiers**. The genome is `encoder_widths`, an
  arbitrary-length list of neuron counts; layer count AND each layer's width come from the
  DNA. The last element is the latent bottleneck; the decoder mirrors it. E.g. `{3,2,1}`
  (256→3→2→1) then `{10,5,4,2}` (256→10→5→4→2). The **one** structural invariant is strictly
  decreasing widths (each layer compresses toward the bottleneck — the defining property of
  an autoencoder), enforced by `repair_widths`. This supersedes phase00's fixed
  `tiny/small/base` tiers — those are three points the free search can reach, not the space.
- `encoding` (SNN only): `direct | latency | poisson`.
- `time_steps` / `voltage_threshold` (SNN only): coupled to `encoding` as phase00 does
  (`direct`→1/1.0; `latency`/`poisson`→16/0.2), not independently varied. `bounds.evolve_temporal`
  can promote them to free genes — a declared expansion.
- training hyperparameters: held fixed per population (see §5.2), not evolved by default.

### 5.1.1 Size of the search space (measured from the code)

**Architecture space — free depth + free per-layer width.** The number of distinct
architectures is the count of strictly-decreasing integer sequences:

```
architectures = Σ_{d=min_layers}^{max_layers}  C(width_room, d),   width_room = max_width − min_width + 1
```

For the shipped bounds (`min_layers=1, max_layers=6, min_width=1, max_width=128`,
`width_room=128`):

| d (layers) | C(128, d) |
|---|---|
| 1 | 128 |
| 2 | 8 128 |
| 3 | 341 376 |
| 4 | ≈ 1.06 × 10⁷ |
| 5 | ≈ 2.64 × 10⁸ |
| 6 | ≈ 5.42 × 10⁹ |
| **Σ (1–6)** | **≈ 5.7 × 10⁹ architectures** |

× SNN encodings (×3) × `n_seeds`. **The architecture space is effectively unbounded relative
to any feasible GA budget**, so the GA is a genuine search — crossover and mutation matter,
and the evaluation cache saves real recomputation rather than papering over an enumeration.

**Categorical axes (`(model, encoding, loss)` triples).** The encoding↔loss pairing is an
enforced invariant (`validate()` throws on a mismatch), so `loss` is bound to `(model,
encoding)`:

| model | encoding | legal `ae_loss_type` | triples |
|---|---|---|---|
| `ann-ae` | `direct` (forced) | `mse`, `mae` | 2 |
| `snn-ae` | `direct` | `mse`, `mae` | 2 |
| `snn-ae` | `latency` | `mse`, `mae`, `spiketime` | 3 |
| `snn-ae` | `poisson` | `mse`, `mae`, `spikecount` | 3 |
| | | **total** | **10** |

10 triples × 4 modality slots (eeg, voice, fused-early, fused-late) = 40, × 4 optimizers =
**160** distinct configuration templates (architecture excluded). Verified against the code:

- **Optimizers — all 4 compatible** (adam, sgd, lion, schedule-free-adamw). `run_protocol_ae`
  forwards `optimizer_type` to `TrainerConfig`; every optimizer honors `attach_with_scales()`
  for the SNN biophysical params (R, C, V_th).
- **Losses — 4 wired, encoding fixes the choice.** `MSELossImpl`, `MAELossImpl`,
  `SpikeCountLossImpl`, `SpikeTimeLossImpl`. `CrossEntropy` excluded (classification). Invariant
  ([Spike Encoding](../Concepts/Spike-Encoding.md)): `direct → mse|mae`, `poisson → spikecount`,
  `latency → spiketime`. `validate()` rejects mismatched pairs, spike losses for `ann-ae`/`lstm-ae`
  or `direct`, and an empty loss (no silent fallback to mse). `mse`/`mae` stay available under
  `poisson`/`latency` as an opt-out baseline.
- **GA consequence:** a spike-loss run must pin `encoding_choices` to the single compatible
  encoding, or mutation can flip the encoding and make `validate()` throw mid-run — the shipped
  `pga_snn_eeg_spike{count,time}.json` do exactly this.

> **Loss dispatch (resolved).** The Trainer's loss is a *compile-time* template parameter
> (`Trainer<ModelType, LossType>`, default `MSELossImpl`) and the old `loss_type` string was
> never read, so only MSE was ever trained. `AutoencoderConfig::ae_loss_type` now dispatches
> to a concrete `Trainer` instantiation in `ThesisFeatureExtraction`.
>
> **Spike-loss silent-failure guard.** A spike loss can emit an all-zero gradient and still
> run to completion (`SpikeTimeLoss` attaches gradient only at the predicted first-spike row;
> a unit that never fires gets nothing). If that holds every batch, the AE trains on nothing.
> It is configuration-dependent — measured, `lr=0.01` was live across 20 seeds while `lr=0.001`
> deadlocked at the same threshold, and firing-rate regularization did *not* rescue it. The AE
> path wraps spike losses in a liveness guard that **throws** when every batch was zero. Treat a
> thrown run as a config to fix (raise lr, lower `voltage_threshold`, raise `time_steps`), not a
> failed individual. See [Spike Encoding](../Concepts/Spike-Encoding.md).
>
> **`spiketime` layout.** `SpikeTimeLossImpl` indexes `t*B + b` — time-major `(T*B, F)`. The
> default AE sample is a single `(1, D)` frame; feeding batched frames would reinterpret samples
> as timesteps. The AE trainer therefore **pre-interleaves** each batch into one `(T*g, D)`
> sample with `row = t*g + b`, honoring `batch_size` via the layout. See
> [What time_steps really means](../Concepts/Time-Steps.md).

### 5.2 Loss functions per population

**ANN:** MSE (primary); MAE available. **SNN:** surrogate-gradient training (Exponential /
Boxcar). `SpikeCountLoss` (poisson) and `SpikeTimeLoss` (latency) are wired and encoding-bound
(§5.1.1). `Van Rossum` / `Victor–Purpura` are future work — they need documented `τ` / `q`
hyperparameters, and Victor–Purpura is expensive (final-front evaluation only, not every
generation).

Log `α`, `β`, `G₁`, `G₂` **per individual**, not just the final `D_penalized`, or degenerate
cases become indistinguishable from ordinary poor ones in post-mortem analysis.

### 5.3 ANN/SNN comparability

`D_penalized` is comparable across technologies (it is computed over `[0,1]`-normalized
feature vectors regardless of what produced them) — the populations meet there. Reconstruction
/ latent-activity is **not** comparable; use per-population `τ_rec` and never compare absolute
values across ANN/SNN. Evolve the two populations **separately** (no cross-population crossover)
and compare only the resulting Pareto fronts.

### 5.4 Signal tracks

Run and report the GA separately per track — `eeg`, `voice`, `fused` (early/late) — mirroring
the thesis phase00/phase01 split. Do not collapse them into one undifferentiated population.
The 12 shipped profiles cover all three tracks for both populations plus the SNN loss variants.

### 5.4.1 Compute budget — sample cap (`dataset.max_samples`)

A full run of one population is at most `population_size × (1 + generations) × n_seeds =
32 × 65 × 3 = 6 240` autoencoder trainings — but the **evaluation cache trains each distinct
expressed phenotype once**, so the real count is far lower (a genome that survives, or that a
different genotype re-expresses, is not retrained). Cost is **linear in sample count**
(measured).

> **Budget note (pop=32, gens=64).** These counts are ≈10× the earlier 16×12 setting. The
> `max_samples` table below was calibrated at 16×12 (~24 h nominal); at 32×64 the *upper bound*
> scales up proportionally, but the cache makes the realized cost strongly sublinear in
> `(1+generations)` once the front stabilises and phenotypes start repeating. Treat the row
> values as per-`(1+generations)×population` and re-measure after the first few generations
> rather than trusting the naive product. If wall-clock matters, the cheapest levers remain
> `n_seeds` (3→1 is a free 3×) then `generations`, before cutting samples.

**Measured per-training cost** (100 epochs, `max-performance` CPU preset): SNN ≈ 5.6 s at 200
samples (averaged over a real 16-genome population, tail included); ANN ≈ 2.2 s. Linear
scaling, whole sweep (5 ANN + 7 SNN profiles):

| `max_samples` | samples/subject (15 subj) | est. total, all 12 (upper bound) |
|---|---|---|
| 200 | 13 | ~8.7 h |
| 300 | 20 | ~13 h |
| 400 | 26 | ~17 h |
| 500 | 33 | ~22 h |
| **550 (shipped)** | **36** | **~24 h** |
| 1974 (full) | 131 | ~86 h |

The estimate is an **upper bound**: objective 2 (inference cost) selects toward cheaper
genomes, so later generations run faster than the random initial population these numbers come
from. Two safety facts:

- **`max_samples` is stratified.** `apply_max_samples` (ThesisDataset.cpp) is round-robin
  across subjects, so a cap keeps balanced samples from *every* subject — no class dropped, α/β
  stay well-formed. Floor: ~15/subject (~225 total) before the metric gets noisy.
- **Cost is dominated by the free-architecture SNN tail** (deep × wide × `time_steps=16`
  genomes). If a run overruns, the cheap extra levers are `n_seeds` (3→1 is a free 3×, losing
  only the stability std) then `generations`, before cutting samples further.

Shipped default: **`max_samples=550`** on all 12 profiles → ~24 h nominal (~36/subject, the
most data that fits the window). Lower toward 300–400 for faster iteration.

### 5.5 Training budget and reproducibility

- Fixed epoch budget per individual with a declared early-stopping criterion.
- Each individual trains under `n_seeds` distinct seeds; its `D_penalized` is the mean, with
  the std recorded as run-to-run stability. This multiplies cost by `n_seeds`.
- Every run is reproducible from a versioned config file (`ga.seed` drives the single RNG that
  seeds init, selection, meiosis, and mutation — the first population is fully random).

### 5.6 Reproduction: true diploid genetics + selection policy

The GA is **diploid**, not haploid. Each individual carries **two** haplotypes (two full
`Genome` copies) plus a dominance value per haplotype; only the haplotype with the larger
dominance is *expressed* — built into an AE and scored. The other rides along silently.

Why bother, when only one haplotype is ever evaluated? Because the silent haplotype is a
**reservoir of alleles that pay no current fitness cost but can resurface later** (Goldberg &
Smith 1987 [66]). A haploid GA forgets every gene the population stops expressing; a diploid GA
keeps a hidden second copy that recombination can promote generations later — cheap protection
against premature convergence, and a natural fit for a search whose fitness landscape (objective
1) is currently weak.

**One generation, end to end** (`GaNsga2.cpp::run_nsga2`, bounds/probs from the `ga` block):

| Step | Policy |
|---|---|
| **Population size** | `population_size = 32` individuals, constant every generation. |
| **Parent selection** | Binary tournament (`tournament_k = 2`) under the crowded-comparison operator (lower rank wins; ties by larger crowding). |
| **No self-mating** | The two parents are always distinct — the second tournament excludes the first winner's index (`tournament_excluding`). |
| **Meiosis** | Each parent recombines its own two haplotypes into one haploid **gamete** — `crossover(hap_a, hap_b)` with probability `crossover_prob = 0.9`, otherwise a straight copy of one haplotype. The gamete then mutates (`mutation_prob = 0.2`, three structural operators: width jitter / add layer / remove layer, plus encoding for SNN). It carries one parent-haplotype's dominance value, itself mutable. |
| **Fusion** | The two gametes fuse into the diploid child (gamete 1 → haplotype A, 2 → B). Reproduction is **always** the fusion of two gametes — there is no clone path. |
| **Elitism (winners)** | μ+λ: parents+offspring (64) are non-dominated-sorted together; the best `population_size − n_losers = 30` fill the next generation by (rank, then crowding). A good solution can never be lost. |
| **Loser reserve** | The remaining `n_losers = 2` slots are filled from the **non-winners**, worst (highest) rank first, ties by larger crowding — deliberately carrying losing genetic material forward as an anti-local-optimum hedge. `n_losers = 0` recovers textbook NSGA-II. |
| **Best-fitted output** | The final feasible rank-0 front, deterministically ordered by (inference cost, then `d_penalized`). |

**Expression / dominance.** `expressed() = dom_a ≥ dom_b ? hap_a : hap_b` (ties → A,
deterministically). The evaluation cache keys on the *expressed* phenotype, so two genotypes
that express the same architecture share one training — the diploid layer adds reproduction
memory without multiplying training cost.

**Failure mode (silent).** If dominance never mutated and gametes never mixed dominance across
haplotypes, the recessive reservoir would be frozen and diploidy would buy nothing — it would
look like it was working while behaving haploid. Guarded by `PgaDiploid.RecessiveAlleleCanResurface`,
which asserts a hidden haplotype can reach expression through meiosis+fusion alone.

---

## 6. Outputs

Written to `results/paraconsistentGA/` per run (`run_tag`):

- `pga_<tag>_individuals.csv` — one row per distinct genome: `born_generation`, `depth`,
  `latent`, `encoder_widths`, `encoding`, `time_steps`, `voltage_threshold`, `α`, `β`, `G₁`,
  `G₂`, `d_truth`, `d_penalized` mean+std, `latent_activity`, `param_count`, `inference_cost`,
  `est_latency_ms`, feasibility, constraint violation.
- `pga_<tag>_pareto.json` — the final feasible Pareto front plus run metadata (population,
  GA params incl. `n_losers` and `ploidy: diploid`, constraints) and an explicit
  UNCALIBRATED-latency warning until the proxy is calibrated. Each front member records both
  its expressed `genome` **and** its diploid `genotype` (`hap_a`, `hap_b`, `dom_a`, `dom_b`,
  `expressed`), so the recessive reservoir behind each winner is traceable.

The per-individual CSV logs the **expressed phenotype** (one row per distinct expressed
genome); the recessive haplotype is genotype-level and appears only in the Pareto JSON.

The versioned config carries all hyperparameters, including `λ`, `τ_rec`, `n_seeds`, epoch
budget, latency ceiling, **optimizer**, and **`ae_loss_type`** (`mse | mae | spikecount |
spiketime`, encoding-bound per §5.1.1) — optimizer and loss are the axes that expand the AE
template space from 16 to 160. (`τ` for Van Rossum and `q` for Victor–Purpura belong here too
once those losses exist.)

### 6.1 Crash resilience — checkpoint & resume

A full sweep runs for days, so a power loss must not throw away hours of training. The problem
is entirely about the **expensive** step: training an autoencoder is seconds each, ×thousands;
everything else is microseconds. So checkpointing protects *trainings*, at two granularities
(`GaCheckpoint.cpp`, controlled by the `checkpoint` config block):

| Layer | File | Written | Guarantees |
|---|---|---|---|
| **1 — per-individual cache** | `pga_<tag>_cache.jsonl` | one line appended + flushed the instant each genome is scored | no genome is **ever** retrained across restarts |
| **2 — per-generation state** | `pga_<tag>_checkpoint.json` | population + exact RNG state + generation index, written atomically (temp-file + rename) after each generation | the loop resumes from the next generation |

**Worst-case loss on a crash = one in-flight training** (seconds). Resume is deterministic
regardless of whether AE training is bit-reproducible, because layer 2 restores the exact
population and RNG engine state rather than replaying past generations; layer 1 then makes every
already-trained genome in the interrupted generation an instant cache hit.

**Concretely:** power dies at generation 50/64. On restart the binary finds the checkpoint,
restores the generation-49 population + RNG, warms the cache from the JSONL (every genome trained
through gen 50 is a hit), and continues at generation 50 — retraining only the handful of gen-50
genomes that had not yet been scored when the power went.

Two scopes cooperate:

- **Within a profile** — the binary auto-resumes from `checkpoint.json` if present.
- **Across the 12 profiles** — the runner skips any profile whose final `pga_<tag>_pareto.json`
  already exists (a completed profile deletes its checkpoint+cache on success, leaving only the
  CSV + Pareto JSON). So a restarted sweep continues at the profile it died on.

`CLEAN_RESULTS=1` wipes everything for a deliberate fresh start; `checkpoint.enabled=false` opts
out (a crash then loses the whole profile). The **silent** failure this defends against: without
it, a mid-sweep power loss looks *identical* to a normal restart — the run just quietly redoes
days of work.

---

## 7. Correctness guarantees

Enforced by `paraconsistent_ga_gtest` and `thesis_freearch_gtest` unless noted:

- No already-existing component was reimplemented (§2).
- `D_penalized` reproduces the reference cases: `(α,β) = (1,1) → 2.0000` (the Ambiguity vertex;
  `λ = 2−√2` is chosen precisely so every non-Truth vertex scores exactly 2 — an earlier draft's
  `2.4142` was wrong); `(α,β) = (0.92, 0.075) → 0.1580`.
- An individual with constant output is ranked **worst**, not best (constant-latent test).
- No infeasible (over-budget or latent-collapsed) individual reaches the final Pareto front —
  guaranteed by constrained dominance.
- Re-running from the same config and seeds reproduces identical results.
- **The GA budget is smaller than the reachable search space** — satisfied by construction now
  that architecture is free (~5.7×10⁹ shapes against a few-hundred-evaluation budget). There are
  **no pre-defined layer configurations**: layer count and per-layer neuron count both come from
  the DNA, and genomes like `{3,2,1}` then `{10,5,4,2}` are both reachable and legal.

## See Also

- [paraconsistentGA](ParaconsistentGA.md) — the didactic overview + data flow
- [Multi-Objective Optimisation (NSGA-II)](../Concepts/Multi-Objective-Optimisation.md)
- [Paraconsistent Logic](../Core/Paraconsistent.md) — the `D_penalized` objective
- [Spike Encoding](../Concepts/Spike-Encoding.md) — encoding↔loss invariant
- [What `time_steps` Really Means](../Concepts/Time-Steps.md) — the `spiketime` layout requirement
- [Thesis](Thesis.md) — the phase00 baseline this search extends

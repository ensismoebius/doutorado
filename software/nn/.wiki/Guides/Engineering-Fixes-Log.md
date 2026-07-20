# Engineering Fixes Log

Why the code and the thesis look the way they do today — a decision/fix log for the batch of
work that landed 2026-07-16 through 2026-07-19, covering the Phase 00 selection metric, the
`snn_lr_scale`/optimizer machinery, network-level parity testing against PyTorch/snnTorch, and
the resulting Thesis/Guayaquil re-run. This is the permanent home of the former repo-root `fixme.md`;
source comments citing **"D1"–"D6"** refer to the decisions below. See the
[Re-run Runbook](./Re-run-Runbook.md) for the commands these fixes made necessary, and
[Experiment05](../Experiments/Thesis.md) for current results.

## Status

| # | Issue | State |
|---|---|---|
| D1 | Phase 00 criterion rewarded a dead autoencoder | ✅ Resolved — `d_penalized` metric |
| D2 | §08 conclusion about temporal encoding was confounded by D1 | ✅ Resolved — claim withdrawn, mechanism quantified instead |
| D3 | `snn_lr_scale` was a global multiplier, not per-parameter-group | ✅ Fixed in code; Guayaquil+Thesis re-run completed 2026-07-19 |
| D4 | Thesis section on per-group learning rate | ✅ Written (§2.1.10.10) |
| D5 | Item 51 (optimizer ablation) | 🟡 Framework done (polymorphic optimizers, Lion, Schedule-Free AdamW, ground truth); **ablation itself not yet run** |
| D6 | EEG `scale` axis is inert (Bark/Mel degenerate to linear) | ✅ Resolved — axis removed for EEG, grid 300→208 |

---

## D1 — The Phase 00 criterion rewarded a dead autoencoder

**Problem.** Selecting the Phase 00 winner by minimum $D_{\text{truth}}$ let a collapsed
("dead") latent — one that emits the same vector for every sample — win. A dead latent lands
exactly on the paraconsistent **Ambiguity** vertex ($\alpha=\beta=1$), scoring
$D_{\text{truth}}=\sqrt2\approx1.41$, low enough to beat genuinely separable extractors. Root
cause: the AE branch saturates $\beta\approx0.99$, collapsing $D_{\text{truth}}$ to a function of
$\alpha$ alone, and $\alpha$ is maximized by a constant output.

**Fix.** A new selection metric, keeping $D_{\text{truth}}$ and adding a penalty on the
**contradiction degree** $|G_2|=|\alpha+\beta-1|$ (whose poles are exactly the degenerate
Ambiguity and Indefinition vertices):

$$D_{\text{penalized}} = D_{\text{truth}} + \lambda|G_2|, \qquad \lambda = 2-\sqrt2 \approx 0.586$$

$\lambda$ is chosen so the three non-Truth vertices (Falsity, Ambiguity, Indefinition) all score
exactly 2.0 — no directional bias between the three degeneracies. A more literal alternative
($D_{0,1}+D_{0,-1}-D_{1,0}$, maximize) was tried and rejected: without penalizing Falsity, it
opens a *new* exploit — the optimizer flees toward Falsity, and the worst-performing encoding
(closest to Falsity) starts "winning." Keeping the full $D_{\text{truth}}$ term is what still
penalizes Falsity.

`d_penalized` is a pure function of $(\alpha,\beta)$, both already recorded per run, so
re-ranking existing results needs no re-execution. See
[Core/Paraconsistent.md](../Core/Paraconsistent.md#selection-metric-contradiction-penalized-truth-distance)
for the implementation and [Experiment05 pitfall 12](../Experiments/Thesis.md#common-pitfalls)
for a related bug this surfaced (a thesis table generator that sorted the wrong direction).

**Files:** `ThesisParaconsistent.{hpp,cpp}`, `ThesisOutput.cpp` (CSV `d_penalized` column, JSON
`best_d_penalized`), `01_thesis_phase00_rank.py`, `02_thesis_apply_winner.py`.

**Dead-latent root cause.** The SNN-AE encoder (`linear:16:leaky, linear:8:identity`) has no
nonlinearity after the second Linear (`identity` is a literal no-op), so with `direct` encoding
(T=1) the LIF loses its recurrent state and becomes a pure threshold test on the first Linear's
output. Measured firing rate at default He-uniform init: **8.3%** (85/1024 neurons), rising
monotonically with weight magnitude — the textbook "No-Spike Problem." The framework already had
two defenses (`firing_rate_reg_lambda`, tdBN), but neither was wired to the autoencoder — only to
the DSNN classifier.

**Fix: firing-rate regularization ported to the AE encoder.** Same math as
`ThesisDsnnClassifier::add_firing_rate_grad` (band penalty
$\lambda(\max(0,r_\text{min}-r)^2+\max(0,r-r_\text{max})^2)$), but the AE's encoder is a
generically-built `Sequential` (not named members), so its LIF layers are located once at
construction via `dynamic_cast`, and gradient injection uses a manual reversed-order backward
loop (`ProtocolSpikingAutoencoder::backward_with_firing_rate_reg`) rather than touching the
shared `Sequential` class. New fields default to `0.0` (inert) — `AutoencoderConfig` (exp03) and
`ThesisConfig::AutoencoderConfig` — so no existing profile changed behavior.

**Validated values, applied to all 18 real `snn-ae` Phase 00 profiles (+ 18 smoke mirrors):**
`firing_rate_reg_lambda=0.5`, `firing_rate_min=0.10`, `firing_rate_max=0.80`. `0.80` matches the
DSNN classifier's default; `0.10` sits above the measured 8.3% native rate so the penalty
actually engages; `0.5` was validated on `direct` at the profile's real learning rate (baseline
$\alpha=0.5$ → $\alpha=0.375$ with regularization on). The same validation could not be repeated
on `poisson`/`latency` ($T=16$) — a verification run was interrupted after ~2h without finishing,
confirming those encodings are themselves an expensive experiment at this scale.

Tests: `ProtocolSnnFiringRateRegularizationInjectsGradientWhenEnabled`,
`...InertWhenLambdaZero` (`AutoencoderRedesign_gtest.cpp`).

---

## D2 — The §08 conclusion about temporal encoding was confounded by D1

D1 and D2 are **independent defects of the same geometric criterion**, not two faces of one bug.
D1: the criterion **rewards** zero variance it shouldn't. D2: the criterion can **punish** real
variance that isn't actually worse learning. Recomputing `d_penalized` over the three encodings
from stored data doesn't change their order (`direct` < `latency` < `poisson`) — expected, since
the $|G_2|$ penalty is near-zero away from the degenerate vertices. **Fixing D1 does not fix D2.**

**The mechanism (theory, not measurement — survives the data being withdrawn).** Each encoding
injects, *before any learning happens*, a noise floor computable in closed form:

- `poisson`: an independent Bernoulli draw per time step → the mean over $T$ frames has variance
  $v(1-v)/T$. At $T=16$, $v=0.5$: $\sigma=\sqrt{0.25/16}=\mathbf{0.125}$ — 12.5% of the full
  $[0,1]$ feature range, per sample, present even with a perfect encoder.
- `latency`: deterministic — the same $v$ always yields the same spike time; only quantization
  error, $\le\frac{1}{2}\cdot\frac{1}{T-1}\approx0.033$ (~4× smaller), identical across
  same-valued samples.
- `direct`: zero noise.

Since $\alpha$ is defined from intraclass **amplitude** (min–max), it is maximally sensitive to
outliers — this noise floor maps monotonically onto $\alpha$ and *predicts* the observed order as
a joint artifact of the metric and the encoder's noise floor, not evidence that temporal codes
carry less speaker information.

**Conclusion.** The claim "negative result for temporal encoding" is **not established** — and
not from mere uncertainty: $\alpha$ is *structurally incapable* of telling "the encoder didn't
learn" apart from "this encoding has an irreducible statistical floor at T=16." Since the floor
falls as $1/T$, this is testable ($T=64$ → variance ÷4 → $\sigma=0.0625$) but **has not been
executed**.

**2026-07-19 update.** The Phase 00 re-run (see [Re-run Runbook](./Re-run-Runbook.md)) confirms
the predicted order (`direct` ≻ `latency` ≻ `poisson`) cleanly in both signals — `poisson`
occupies the bottom 3 ranks with $\alpha\le0.003$ across all 6 profiles. This is exactly what the
noise-floor mechanism predicts *a priori*, so **confirming the prediction does not establish the
interpretation** — the $T=64$ test above remains the way to actually distinguish the two
hypotheses.

---

## D3 — `snn_lr_scale` was a global multiplier, not per-parameter-group

The parameter promised a per-group learning rate (smaller for R, C, V_th than for weights), but
in practice was a global multiplier:

```cpp
// Trainer.hpp:100 (before the fix) — fills ALL params, not just R/C/V_th
std::vector<float> scales(params.size(), cfg_.snn_lr_scale);
```

**Scope — larger than first thought.** Not just the 24 Thesis autoencoder profiles: `GuayaquilTraining.cpp`'s
SNN branch calls `make_trainer_config` with `snn_lr_scale=0.1F`, and this gets **overwritten**
whenever a profile sets `learning_rate_biophysical`. The three Guayaquil paper profiles
(`article-snn-{dense,conv1d,recurrent}.json`) declare `learning_rate_biophysical=0.0001` against
`learning_rate=0.001` — the exact same 0.1 ratio as the Thesis bug. The paper's SNN-vs-LSTM table
therefore compared an LSTM baseline trained at 1e-3 against SNN variants training **every
weight** at 1e-4 — a structurally unfair comparison against the paper's own contribution, and a
more serious instance than the Thesis bug: that one affects 24 profiles of an in-progress thesis;
this affected a table in an already-drafted paper.

Full scope: 24 Thesis AE profiles (6 ANN + 18 SNN — ANN too, since it shares the same `Trainer`
default; only handcrafted extraction, which trains nothing, escapes), the 3 Guayaquil `article-snn-*`
profiles, and (behaviorally, though with no profile ever having produced results before the fix)
the Thesis DSNN classifier.

**Fix.** Rather than tagging biophysical parameters in the `Module` contract (which would change
`params()`'s signature everywhere), the fix exploits an already-true fact: R, C, V_th are always
1×1 scalar tensors (`LifImpl`/`LifBPTTImpl`), while weights/biases never are — the same size
criterion `Adam`'s `weight_decay` already uses to exclude biases/biophysicals from decay:

```cpp
std::vector<float> scales(params.size(), 1.0F);
for (std::size_t i = 0; i < params.size(); ++i)
    if (params[i]->size() == 1) scales[i] = cfg_.snn_lr_scale;
```

No interface change. Regression test: `trainer_genericity_gtest.cpp`,
`SnnLrScaleOnlyAppliesToSizeOneParams` — a model with one 1×1 and one 2×2 parameter, same fixed
gradient, confirms only the 1×1 moves by `lr·snn_lr_scale` after one Adam step.

**Re-run status: completed 2026-07-19.** Both the Guayaquil SNN profiles and the Thesis Phase 00/01 grids
have been re-executed under the fix — see [Re-run Runbook](./Re-run-Runbook.md) and
[Experiment05](../Experiments/Thesis.md#overview) for results.

---

## D4 — Thesis section on per-group learning rate

Written in `chapters/07-bibliographicRevision.tex` §2.1.10.10 ("Taxa de aprendizado por grupo de
parâmetros"), deliberately *after* D3 was fixed — writing it earlier would have documented an
intention the code didn't honor. Covers: why R/C/V_th need a smaller lr than weights; an explicit
honesty note that the 0.1 factor is a project engineering choice, not a literature value (a prior
citation for this claim was removed as unverifiable); the real Adam mechanism
(`attach_with_scales`); the original defect and how it was found (during the dead-latent
investigation, D1); the actual blast radius including the Guayaquil paper; the size-based fix;
the regression test; and the open re-run status at the time.

---

## D5 — Item 51: optimizer ablation infrastructure

**Original blocker.** No optimizer selection existed anywhere in Thesis — `Trainer` held a concrete
`Adam optimizer_` member, no config field selected an optimizer, and `attach_with_scales` was
Adam-only (non-virtual, absent from SGD).

**1. Per-group scales moved to the base class.** `attach_with_scales()` became `virtual` on
`Optimizer`, with a default implementation that calls the (also virtual) `attach()` and then
stores the scales. `Adam`, `SGD`, and `SGDMinimal` all now read `lr_scales_` in `step()` and apply
decoupled `weight_decay` under the same 2-D-matrix-only restriction. Before, only Adam honored
scales — SGD/SGDMinimal silently ignored any passed in, which is exactly D3's bug class.

**2. `Trainer` made polymorphic.** `std::unique_ptr<::Optimizer> optimizer_`, built by
`OptimizerFactory` from two new `TrainerConfig` fields: `optimizer_type` (default `"adam"` —
identical behavior for every existing caller/profile) and `optimizer_momentum`. Unknown type
throws rather than silently falling back to Adam.

**3. A latent bug found and fixed in passing.** `Optimizer::attach()`'s base implementation was a
literal no-op, despite its own comment claiming concrete optimizers should call it "to preserve
this storage for no-arg convenience methods." Adam worked around this by setting
`attached_params_` itself; **SGD trusted the contract and never populated it, so `sgd.step()`
with no arguments always threw**, even after a correct `attach()`. The base now honors what it
documents.

**4. SOTA optimizers: Lion and Schedule-Free AdamW.** Ported by reading the reference
implementations' source (not from memory — precedent: a fabricated citation found in item 57
below made this the deliberate policy). This caught details a "from memory" port would miss:
Lion applies decay *before* the update and advances momentum *after*; Schedule-Free maintains
three coupled sequences (x/y/z) and *evaluates* at a different point than it *trains* at.
`Optimizer::train_mode()` + a RAII `OptimizerEvalScope` were added so `Trainer` validates at
Schedule-Free's averaged iterate `x` (no-op for the others).

**Rejected, with concrete technical reasons:**
- **Muon** — implemented and parity-validated, then removed at the author's request: it
  orthogonalizes only genuine 2-D matrices (`rows>1 && cols>1`) and falls back to Adam for
  everything else — since R, C, V_th are 1×1 here, it would never touch them, and its target
  use case (LLM-scale pretraining) has no plausible role in this ablation. Recoverable from git
  history.
- **Sophia** — incompatible with the current `Optimizer::step(span<Tensor*>)` contract; its
  Hessian estimate needs a *second backward pass* with resampled labels
  (Gauss-Newton-Bartlett), which `Trainer`'s loop has no hook for. Implementing it without that
  would silently degrade to "Hessian = squared gradient," effectively relabeling Adam's second
  moment as Sophia.
- **SOAP** — needs eigendecomposition (`eigh` of $GG^\top$/$G^\top G$), absent from the
  backend-agnostic `Tensor` interface; adding it would mean extending the
  `TensorBackendParityContract` across all four backends (XTensor, OpenCL, SYCL, Device), and
  only XTensor has `xtensor-blas` to cover it.

**Ground truth.** `scripts/testing/gen_optimizer_refs.py` drives the real reference
implementations (`torch.optim.{Adam,AdamW,SGD}`, `lion-pytorch`, `schedulefree`) over a fixed
param/grad sequence and records the parameter after each step; `optimizer_parity_gtest` (10/10
green) replays the same data through this port.

> **Scale warning for the ablation.** Reference lr defaults diverge sharply (Adam 1e-3, Lion
> 1e-4, Schedule-Free 2.5e-3), and Lion's update is `±lr` on *every* coordinate regardless of
> gradient magnitude, so its usable lr is much smaller. **A single shared lr is not a fair
> ablation** — it measures the lr choice, not the optimizer.

**Two real bugs the ground truth found:**
- **Adam's weight decay was applied in the wrong order.** We applied it *after* the gradient
  step; Loshchilov & Hutter (ICLR 2019) define it against $\theta_{t-1}$, and
  `torch.optim.AdamW` matches that order. Confirmed numerically: decay-before matches torch to
  **0.0** error; decay-after errs by 1e-5. Fixed in Adam and, for consistency, SGD/SGDMinimal
  (no ground truth existed for those two — `torch.optim.SGD`'s decay is coupled L2, not
  decoupled).
- **A `nn::Tensor` value-semantics trap.** Moving decay before the update broke Adam's updates
  entirely: assigning to `param` replaces its storage and discards the gradient buffer (a trap
  already documented in `SGDMinimal.hpp`), so reading `param.grad()` after decay read an empty
  gradient. Fixed by saving/restoring the gradient around the decay step.

**5. Each profile now resolves the learning rate of its own optimizer.** Risk: today all 333
profiles use `adam` + `lr=0.001`, which happens to be Adam's own reference default, so nothing is
currently wrong — but the moment an ablation profile sets `optimizer_type: lion` and leaves
`lr: 0.001`, Lion trains 10× too hot and the conclusion would be "Lion is worse," a D3-shaped
false finding. Fixed structurally: `nn::optimizers::reference_learning_rate(token)` is the single
source of truth for each optimizer's published default; `training.learning_rate` became
`std::optional` in the profile schema — omitted → resolved from the chosen optimizer; declared →
still wins (lr sweeps remain possible). The summary now records what actually ran: a `training`
block with `optimizer_type`, the **resolved** `learning_rate`, and `learning_rate_source`
(`profile` | `optimizer_default`) — closing exactly the provenance gap that let D3 go unnoticed
for as long as it did.

**Still pending: the ablation itself has not been run.** No real profile declares a non-`adam`
`optimizer_type` yet. Running it requires (a) deciding whether the per-optimizer reference lr
suffices or each optimizer needs its own sweep, and (b) machine time — this is an expensive
experiment under the same guard as D1–D3.

**Files:** `Optimizer.hpp`, `Adam.hpp`, `SGD.hpp`, `SGDMinimal.hpp`, `Lion.hpp`,
`ScheduleFreeAdamW.hpp`, `OptimizerFactory.hpp`, `Trainer.hpp`, `TrainerConfig.hpp`,
`ThesisConfig.{hpp,cpp}`, `ThesisClassifiers.cpp`, `ThesisFeatureExtraction.cpp`,
`scripts/testing/gen_optimizer_refs.py`, `optimizer_parity_gtest.cpp`. Docs:
[Core/Optimizers.md](../Core/Optimizers.md), [Core/Training.md](../Core/Training.md).

---

## D6 — The EEG `scale` axis is inert

**Confirmed, and more thoroughly than first suspected.** Checked against all 300 stored results
rather than trusting the earlier note: on EEG, the three scales give **bit-identical** `d_truth`
in **46/46** wavelet×category groups (2 apparent daub32 exceptions were ~1.5e-6 floating-point
noise from an individual profile re-run, not a scale effect). On voice, all 138 groups genuinely
differ, as expected.

**The originally recorded cause was wrong.** It claimed EEG content sits "below" the band
structure so grouping collapses. Two things wrong with that: (1) `group_by_scale()` normalizes
by the signal's *own* Nyquist frequency, so the curve stretches to fit any sample rate — nothing
is fixed to fall "below"; (2) nothing collapses — the opposite happens. EEG's 16 sub-bands land in
**16 distinct bins** (nothing merges); it's *voice* that merges (bark→9 groups, mel→11).

**The real cause.** Bark is an *absolute* scale — ~24 Barks span the audible range, and
`n_bands=24`. For voice the normalization factor is ~0.97 (a near no-op). For EEG (Nyquist 512 Hz)
the factor is **4.96**: the curve is stretched 5× to fill the same 24 bins. Since Bark/Mel are
approximately linear over that stretched range, the mapping becomes injective — identical to
linear. **"Bark" on EEG was never actually Bark** — it was a linearly-rescaled pseudo-scale. The
normalization was designed for audio, where it's harmless, and misbehaves at any other sample rate.

**This had already contaminated a reported result.** The EEG Phase 00 winner had been reported as
`haar_bark_c1` — but `haar/lfcc/c1` and `haar/mel/c1` score *exactly* the same. The three were a
perfect tie, and "Bark won" was purely an artifact of sort tie-breaking; Bark did nothing, and the
winning vector was literally the linear one.

**Fix (author's decision: remove the `scale` axis for EEG entirely, on physiological grounds —
not merely because it's redundant):**
- 92 `p00_hc_*_{bark,mel}_*_eeg.json` profiles removed (+ 92 smoke mirrors, 184 files, all
  git-tracked). Grid: **300 → 208** (46 handcrafted EEG + 138 handcrafted voice + 24 AE). No
  re-run needed — the existing `lfcc` results already are the answer.
- `ThesisConfig::validate()` now rejects `handcrafted.scale != "lfcc"` when `modality=eeg`. `fused`
  is deliberately unrestricted (voice's half legitimately uses bark/mel). Test:
  `ThesisEegScaleAxis.BarkAndMelAreRejectedForEeg`.
- `01_thesis_phase00_rank.py` now detects exact ties (1e-5 tolerance, absorbing daub32 float noise)
  and records `tie_count`/`tied_with` in `winners.json`, printing an explicit `[TIE]` warning.
- `extractor_label()` now includes the cepstral category (`c1`/`c2`) and, for autoencoders, the
  encoding and latent size — it previously omitted these, making `haar/lfcc/c1` and
  `haar/lfcc/c2` print identically as duplicate-looking rows.
- Thesis: 4 stale counts fixed (138/signal → 138 voice + 46 EEG; 300 → 208), new subsection on
  Bark/Mel applicability to EEG (§2.1.3.9, `sec:escalaEeg`).
- `thesis_build_phase00_paraconsistent_tables.py`: explicit filter (`[info] skipped N retired EEG
  bark/mel run(s)`) so the 276 orphaned result files from the retired profiles don't still
  produce duplicate rows.

**Phase 00 results were deleted 2026-07-16** (author's decision) rather than kept stale — they
were invalidated by three same-day fixes at once (D1, D3, D6) plus the Adam weight-decay order
fix, which affects any trained model. The 184 handcrafted result files were not numerically
obsolete (handcrafted extraction trains nothing, so a re-run reproduces them bit-for-bit) but
were cleared anyway so the whole phase regenerates as one coherent set whose summaries carry the
new provenance-closing `training` block. **Superseded 2026-07-19**: both phases have since been
fully re-run — see [Re-run Runbook](./Re-run-Runbook.md).

---

## Network-level parity testing vs. PyTorch/snnTorch

Requested before any re-run was launched, to catch composition-level bugs that per-layer parity
(`pytorch_parity_gtest`) can't: a layer can be individually correct while the network built from
it is wrong (gradient chaining, state leaking across sequences, gate ordering visible only in
composition). New: `scripts/testing/gen_micro_network_refs.py` → committed `.npz` (CI needs no
torch) + `micro_network_parity_gtest` (8/8 green).

**Three real bugs found — the test paid for itself before any experiment ran:**

1. 🔴 **`MSELoss`/`MAELoss` silently clipped their own gradient at norm 1.0**, unconditionally and
   non-configurably (`kMaxGradientNorm = 1.0F`, fixed). `MSELossImpl` is `Trainer`'s **default**
   loss, so this hit every autoencoder ever trained in the project, including all 24 Thesis AE
   profiles and the Guayaquil models. It also directly contradicted the caller's own
   configuration: `TrainerConfig::grad_clip_norm` defaults to `0.0` ("no clipping"), and the
   clip fired underneath it regardless. Since it only triggers above norm 1, the effective
   learning rate became a nonlinear function of gradient magnitude rather than a constant
   rescale. `CrossEntropyLoss`/`SpikeCountLoss`/`SpikeTimeLoss` did *not* do this — the losses
   were mutually inconsistent. Same defect class as D3: declared behavior ≠ executed behavior.
   **Fix: made configurable, default OFF** (`max_gradient_norm = 0.0F`) — `backward()` now
   returns the exact gradient by default, matching torch to the decimal; explicit clipping
   remains available as an escape hatch.

2. 🟡 **The LIF layer is not "exactly snnTorch's `snn.Leaky`" — only in the mode this project
   uses.** A prior comment claimed exact equivalence; false for `reset_mechanism="subtract"`. Our
   reset applies **immediately** (`v -= V_th`, decaying starting next step); snnTorch's is
   subtracted **un-decayed** the following step. Measured: 1.9–3.0% spike disagreement in
   `subtract` mode, **0.0%** in `zero` mode. **Real-world impact: none** — `Lif`/`LifBPTT` default
   to `reset_zero=true`, and no production code path selects `subtract`, so the thesis uses
   exactly the mode that matches perfectly. Neither implementation is "wrong" — the *equivalence
   claim* was. Why it went unnoticed: the pre-existing per-layer fixture only triggers reset 3/36
   times (8%), too weak to exercise this path. Fixed by asserting both modes explicitly under a
   strong drive; the divergence in `subtract` mode is intentionally left as a documented,
   unused-path discrepancy — aligning it would require inverting the stored post-reset state to
   pre-reset in both forward *and* the `LifBPTT` BPTT backward (which chains `v_post_history`
   into `dL/dR`, `dL/dC`, and the reset term of `dL/dV_th`) — surgery on the neuron the entire
   thesis depends on, to fix a path nothing selects.

3. 🟡 **The LSTM does not use exact sigmoid/tanh.** It uses `FastActivations.hpp`'s rational
   (softsign) approximations by default, chosen for speed. Not close: $|\tanh - \text{rat\_tanh}|$
   reaches **0.306** over $[-4,4]$ (at $x=2$: tanh=0.964 vs. ours=0.667), and measured hidden-state
   divergence against `torch.nn.LSTM` is **0.1626**. So this is a genuinely different
   "softsign-gated LSTM," not directly comparable to a standard LSTM without a tolerance loose
   enough to prove nothing. Relevant to the Guayaquil paper, which compares "LSTM-AE" against SNN.

**Deliberate, documented scope limits:** spiking backward isn't compared (our surrogate is
exponential, snnTorch's is arctan — different functions by design), so backward parity is
checked only in **readout mode** (no surrogate, both sides must match exactly); forward *with*
spikes is compared normally. A separate test confirms `reset_state()` actually isolates
sequences (SNN invariant #4, see `CLAUDE.md`).

## Reference-fidelity-by-profile: matching PyTorch/snnTorch exactly

Follow-up decision, in four parts: (1) profiles should be able to opt in/out of the fast
approximations; (2) gradient clipping should be profile-configurable, default OFF; (3) the code
should also match snnTorch's `subtract` reset mode (documented above, kept as a known
divergence); (4) since PyTorch/snnTorch behavior is the reference, defaults should match it
wherever there was a choice.

- **`training.gradient_clip_norm`** (default `0.0` = OFF), plumbed to `TrainerConfig::grad_clip_norm`
  in all three places Thesis builds a `Trainer`. Combined with the MSELoss/MAELoss fix above, there
  is now **one** clipping knob, visible in the profile, instead of a hidden one silently
  overriding it.
- **`numerics.exact_activations`** (default `true`). Exact sigmoid/tanh became the default;
  the softsign approximation is now explicit opt-in. `LSTMLayer` gained `exact_activations`
  (default `true`), honored in both forward and backward.
  > **Result: the LSTM now matches `torch.nn.LSTM` exactly**, element-for-element, on every
  > backend. `PyTorchParityTyped.LSTMLayerForward` used a loose 0.25 bound specifically because
  > the layer was unconditionally softsign; it is now a tight comparison.
- 🔴 **A second real bug found while implementing exact activations: the LSTM backward didn't
  match its own forward.** The backward computed $y(1-y)$ and $1-y^2$ — the derivatives of the
  *exact* sigmoid/tanh — while the forward ran the rational approximations. In fast mode the
  gradient was the derivative of a function the forward never evaluated, wrong by up to **5×**
  (at $x=2$: $1-y^2=0.556$ vs. the correct $\text{rat\_tanh}'(2)=0.111$). Making exact the
  default fixes this for free; for fast mode the closed forms were re-derived from the cached
  output ($s=y-0.5$: $\text{rat\_sig}'=(1-2|s|)^2/2$, $\text{rat\_tanh}'=(1-|y|)^2$), verified
  against analytical derivatives to ~1e-16 error.

**Files:** `FastActivations.hpp`, `LSTMLayer.hpp`, `MSELoss.hpp`, `MAELoss.hpp`, `Lif.hpp`,
`LifBPTT.hpp`, `ThesisConfig.{hpp,cpp}`, `ThesisClassifiers.cpp`, `ThesisFeatureExtraction.cpp`,
`pytorch_parity_gtest.cpp`, `micro_network_parity_gtest.cpp`, `gen_pytorch_refs.py`,
`gen_micro_network_refs.py`.

---

## Untrustworthy-comment audit

A targeted (not exhaustive) sweep for the defect class that had already bitten this project
twice: a comment asserting a contract the code doesn't honor (D3, D5), and vague/wrong citations
(item 57 below).

**Fixed:**
- `Optimizer::attach()`'s comment claimed concrete optimizers must call the base method "to
  preserve this storage" — the body was a no-op that preserved nothing (see D5's SGD bug).
- `TrainerConfig.hpp`'s "SNN-specific fields are ignored for pure ANN models" was false before D3
  (it scaled all weights of any model) and became true again *because of* the D3 fix, not an
  edit to the comment.
- `Trainer.hpp`'s changelog listed "snn_lr_scale wired via attach_with_scales (was silently
  ignored)" as a fixed bug — but presented, as the fix, exactly the line that *was* the D3 bug
  (wiring existed, but wired wrong). Rewritten with the real behavior and an explicit note that
  the old wording was misleading.
- `Lif.hpp` misattributed a citation ("[34-35] MPD-ATP; AR-LIF") — `[35]` (MPD-ATP, Wang et al.,
  IEEE Xplore 2025) was correct, but `[34]` is Lv et al. (PMC 2025) on spatiotemporal adaptation,
  not an arXiv paper called "AR-LIF."
- `Adam.hpp`'s weight-decay comment described the wrong order — the code was wrong, the comment
  accurately described the wrong code (see D5).

**Found, deferred as low severity:**
- 106 `@file` comments pointed at nonexistent paths (41 by an earlier, narrower count) — fixed
  each in the style it already used (full-path claims → real path; basename claims → real
  basename), rather than uniformizing all 350 files and inflating the diff.
- `LifBPTT::spike_history` — dead field, single reference (its own declaration), not touched by
  `state_dict`/`reset_state`. Removed; compiles clean.

---

## Pending / open items

- **Item 51 — optimizer ablation.** Infrastructure complete (see D5); the ablation run itself is
  not scheduled. Needs a per-optimizer lr grid decision plus machine time.
- **`training.weight_decay > 0`, `training.firing_rate_reg_lambda > 0`, tdBN** — all implemented
  and validated in isolation, but never exercised end-to-end outside `debug.json`/smoke profiles.
  No real Phase 01 profile enables any of the three.
- Guayaquil-style rich run diagnostics for the thesis — done, see the "What to expect" section of
  the [Re-run Runbook](./Re-run-Runbook.md).

## Deferred / rejected

- Step-by-step wavelet-packet transform derivation in the thesis — declined.
- Cataloging feature-consistency techniques beyond paraconsistent engineering — declined for now.

## Historical resolved issues

- **Voice+EEG fusion bug (2026-07-03).** The wiki described `modality=fused` as concatenating
  voice+EEG feature vectors; the code actually picked audio-if-present-else-EEG — a single
  signal, no fusion at all (neither early nor late). Fixed: real late fusion (independent
  per-signal extraction, vectors concatenated) and early fusion (raw signals concatenated before
  a single extraction pass, `fusion_mode` config field) both implemented; discussed as an
  experimental axis in the thesis and wiki. 3 new tests (`ThesisFusion.*`).
- **4 daub32 profiles failing in a batch run** — transient resource contention from parallel
  workers, not a code defect; passed cleanly on individual re-run.
- **`tempStrategy.tex` accidentally compiled into the thesis** — a scratch file (its own header
  said "not `\input` anywhere") was nonetheless `\include`d as the very first pretextual element.
  Removed (page count 118→111); its reference material was migrated into this log and the
  [Re-run Runbook](./Re-run-Runbook.md).

## Related

- [Re-run Runbook](./Re-run-Runbook.md) — the commands these fixes made necessary
- [Experiment05](../Experiments/Thesis.md) — current design and results
- [Core/Paraconsistent.md](../Core/Paraconsistent.md) — `d_penalized` implementation (D1)
- [Core/Optimizers.md](../Core/Optimizers.md) — polymorphic optimizers, Lion, Schedule-Free AdamW (D5)
- [Running Experiment05 Profiles](./Running-Thesis-Profiles.md) — day-to-day operation

# Training

The `Trainer` class is what actually runs a training run: it takes a model, a
dataset, and a configuration, and repeatedly runs forward passes, computes the
loss, runs backward passes, and updates the weights — while reporting progress
and (optionally) stopping early if the model stops improving. This page covers
the training loop itself, its configuration options, and the progress-bar and
callback system that reports on it.

## Theoretical Background

### The training loop

Training a neural network means repeating this cycle, over and over, once per
batch of data:

1. **Forward pass**: run the current inputs through the model to get its
   current predictions, $\hat{y} = f(x; \theta)$ (read: "the model $f$, with
   its current weights $\theta$, applied to input $x$").
2. **Loss**: compute a single number, $L(y, \hat{y})$, that measures how wrong
   those predictions were compared to the true answer $y$.
3. **Backward pass**: compute the gradient $\nabla_\theta L$ — for every
   weight in the model, how much (and in which direction) changing that
   weight slightly would change the loss.
4. **Update**: nudge every weight a little in the direction that reduces the
   loss: $\theta \leftarrow \theta - \eta \nabla_\theta L$, where $\eta$ is the
   learning rate (see [Optimizers](./Optimizers.md) for what happens here in
   more sophisticated optimizers than plain gradient descent).

### Gradient clipping

Sometimes, especially in recurrent networks like LSTMs, the gradient computed
in step 3 above can become extremely large — large enough that the update in
step 4 overshoots wildly and destabilises training instead of improving it.
**Gradient clipping** guards against this by shrinking the gradient (without
changing its direction) whenever its overall size exceeds a chosen limit
`max_norm`:

$$\nabla_\theta L \leftarrow \min\left(1, \frac{\text{max\_norm}}{|\nabla_\theta L\|}\right) \nabla_\theta L$$

In plain terms: if the gradient's size is already under the limit, leave it
alone; if it's over, scale it down until it exactly reaches the limit.

### Mini-batch training

Rather than computing the loss and gradient over the *entire* training set
before every single update (which would be extremely slow, and requires
holding the whole dataset's activations in memory at once), training
typically processes a small random subset — a **mini-batch** — at a time.
This has two benefits: it's much cheaper per step, and the randomness of
which samples end up in each batch acts as a mild source of noise that
actually *helps* the optimizer avoid getting permanently stuck in a poor
local solution.

## Progress Tracking

Long training runs need visible, live feedback — but printing progress
information from inside a tight inner loop can itself slow that loop down if
done carelessly. This project's progress-bar system is built to avoid that: it
runs the actual screen-drawing on a background thread, so calling `update()`
from the training loop is cheap and never blocks waiting for the terminal to
redraw.

### ProgressBar (a self-contained handle)

Creating a `ProgressBar` registers a new bar to be drawn. Its destructor is
deliberately a **no-op** — despite the name, this is *not* an RAII handle that
auto-removes the bar on scope exit (that would flicker the display for
short-lived handles); call `ProgressManager::instance().remove_bar(id())`
explicitly if a bar needs to disappear before the run ends:

```cpp
// File: include/progress/ProgressBar.hpp
namespace nn::progress
{
class ProgressBar
{
public:
    ProgressBar(const std::string& label, float target);
    ~ProgressBar();  // intentional no-op — does NOT remove the bar

    void update(float value, const std::map<std::string, float>& metrics = {});
    void mark_complete();   // delegates to ProgressManager::complete_bar(id)

    auto id() const -> uint32_t;
};
}
```

### ProgressManager (the shared renderer behind the scenes)

A single, shared `ProgressManager` actually owns the background rendering
thread and knows about every currently-active bar — `ProgressBar` itself is
just a lightweight handle into it:

```cpp
// File: include/progress/ProgressManager.hpp
namespace nn::progress
{
class ProgressManager
{
public:
    static ProgressManager& instance();

    uint32_t create_bar(const std::string& label, float target);
    void update_bar(uint32_t id, float value, const std::map<std::string, float>& metrics = {});
    void complete_bar(uint32_t id);   // NOT "mark_complete" — that name lives on ProgressBar
    void remove_bar(uint32_t id);
    void shutdown();

    // Per-bar metadata shown alongside the bar (fold number, loss type, ...)
    void set_description(uint32_t id, const std::string& description);
    void set_fold_info(uint32_t id, int fold_number, int total_folds);
    void set_loss_type(uint32_t id, const std::string& loss_type);
    void set_test_loss(uint32_t id, float test_loss);

private:
    ProgressManager();
    ~ProgressManager();
};
}
```

### Design properties

- **Non-blocking**: rendering happens on a background thread, so it never
  stalls the actual training loop waiting for the terminal.
- **Multi-track**: several bars can be active and updating at once (e.g. one
  per model being trained in a comparative experiment).
- **Strictly linear**: progress only ever goes from 0% up to 100%, never
  backward.
- **Push-based**: bars update when `update()` is called, not by polling on a
  timer.
- **No external dependencies**: implemented directly with ANSI terminal escape
  codes, rather than a third-party terminal UI library.

### Example output

```cpp
// Example output:
// LSTM Training   [===================           ] 67% | loss: 0.4523
// SNN Training    [======================        ] 88% | loss: 0.2314
```

## How It Is Implemented Here

### Trainer configuration

```cpp
// File: src/core/training/TrainerConfig.hpp
namespace nn::training
{
struct TrainerConfig
{
    int epochs = 10;
    float learning_rate = 0.001F;

    // Which Optimizer the Trainer builds (via OptimizerFactory):
    // "adam" | "sgd" | "lion" | "schedule-free-adamw".
    std::string optimizer_type = "adam";
    float optimizer_momentum = 0.0F;  // SGD only; ignored by the others

    // Adam parameters
    float adam_beta1 = 0.9F;
    float adam_beta2 = 0.999F;
    float adam_epsilon = 1e-8F;

    // Linear learning-rate warm-up: step k of each fit uses
    // learning_rate * min(1, k / lr_warmup_steps). 0 = off. Adam only (others throw).
    int lr_warmup_steps = 0;

    // Gradient clipping
    float grad_clip_norm = 0.0F;

    // Decoupled L2 weight decay (AdamW). 0 = disabled. Applied only to 2-D
    // weight matrices; biases and SNN scalars (R, C, V_th) are excluded [Loshchilov2019].
    float weight_decay = 0.0F;

    // Batch
    int batch_size = 1;
    unsigned int sampler_shuffle_seed = 42;

    // SNN-specific: per-group learning rate for biophysical parameters (R, C, V_th).
    // Effective SNN lr = learning_rate * snn_lr_scale.
    // 0.1 (10x smaller than the weight lr) is this project's own empirical
    // default, not a figure taken from a published paper.
    float snn_lr_scale = 0.1F;

    // Nested k-fold cross-validation (0 = disabled, plain k-fold).
    // Set nested_cv_outer_folds > 0 for unbiased hyperparameter evaluation [41].
    int nested_cv_outer_folds = 0;
    int nested_cv_inner_folds = 5;
};
}
```

**Choosing an optimizer.** `optimizer_type` names which optimizer
implementation to use; the `Trainer` constructor builds the actual object
through `OptimizerFactory` and stores it as a generic `Optimizer` pointer, so
the training loop's code doesn't need to know or care which concrete
optimizer is behind it — this makes it straightforward to add a new optimizer
without touching the training loop itself. The default, `"adam"`, matches
what every existing caller was already getting before this flexibility was
added; naming an unrecognised optimizer throws an error immediately, rather
than silently falling back to something else. See [Optimizers](Optimizers.md).

**Why SNN parameters get their own, smaller learning rate.** The spiking
network's biophysical parameters (R, C, V_th) control the neuron's firing
threshold and its membrane dynamics directly — a large update to one of them
can push it into an invalid region that the code has to clamp
(see [Membrane Dynamics](../Concepts/Membrane-Dynamics.md)), which destabilises
the derived time constant $\tau = R \cdot C$. Setting `snn_lr_scale = 0.1`
gives these parameters roughly one-tenth the learning rate of ordinary
weights (≈1e-4 when the global rate is 1e-3) — again, this ratio is this
project's own empirically-chosen default, not a number taken from a specific
paper. The `Trainer` constructor passes this scale to
`Optimizer::attach_with_scales()`, which applies the reduced rate **only to
parameters whose `size()` is exactly 1** — R, C, and V_th are always 1×1
tensors, and no ordinary weight or bias tensor ever is, so this rule reliably
targets just the biophysical parameters. (An earlier version of this code
filled the scale uniformly across *every* parameter, which had the
unintended side effect of making `snn_lr_scale` throttle the learning rate of
the *entire* network by 10×, not just the SNN parameters — a bug that's now
covered by a dedicated regression test,
`TrainerGenericity.SnnLrScaleOnlyAppliesToSizeOneParams`.)

**Weight decay.** Setting `weight_decay > 0` turns on decoupled L2
regularisation (see [Optimizers](Optimizers.md#decoupled-weight-decay-adamw--sgdw)
for what "decoupled" means and why it matters). The `Trainer` constructor
simply forwards this value to `Optimizer::weight_decay`, which shrinks only
ordinary 2-D weight matrices by `lr·weight_decay·θ` after every step — biases
and the SNN's biophysical scalars are exempt, so their values aren't pulled
toward zero by a mechanism meant for weight matrices.

**Learning-rate warm-up (`lr_warmup_steps`, added 2026-11-03).**

*The problem.* A freshly initialised network has no good descent direction yet, but a constant
learning rate applies its full size from the very first step. For most layers that is fine. For
a **post-norm** Transformer it is not: Xiong et al. (ICML 2020) [84] show its gradients near the
output are largest at initialisation, so full-size Adam steps at step 1 can throw it into a bad
region it never leaves. Warm-up starts the rate small and lets it grow.

*The rule.* Optimizer step $k$ (1-based) uses

$$\eta_k = \eta \cdot \min\!\left(1, \frac{k}{W}\right), \qquad W = \texttt{lr\_warmup\_steps}.$$

*With meeting01's numbers.* $\eta = 10^{-3}$, batch of 1 window, 1000 training windows, so
1000 steps per epoch. $W = 4000$ (Vaswani et al.'s value [74]) ramps over the first 4 epochs:

```
step k        1      1000    2000    4000    4001 ... 30000
lr       2.5e-7    2.5e-4  5.0e-4  1.0e-3  1.0e-3 ... 1.0e-3
epoch         1         1       2       4       5 ...    30
```

*What it is not.* It is not a decay schedule: after step $W$ the rate stays at $\eta$. It counts
**optimizer steps**, not epochs or samples. With mini-batches of 32 the same $W$ lasts 32× more
epochs.

| | Off (`0`, the default) | On (`W > 0`) |
|---|---|---|
| Step 1 rate | $\eta$ | $\eta / W$ |
| Step $W$ onward | $\eta$ | $\eta$ |
| Behaviour of a run that never sets the field | unchanged, byte for byte | — |

*Mechanics.* `Trainer` calls `apply_lr_schedule()` before every `optimizer_->step(...)` in all
three fit loops, and resets its step counter at each `on_train_begin`, so every `fit` warms up
again from step 1: a second `fit` on the same `Trainer` does **not** continue the previous ramp
(`TrainerGenericity.LinearWarmupRampsTheRateAndRestartsEachFit`). The rate is set through
`Optimizer::set_learning_rate`. Only `Adam` overrides it; on any other optimizer the base-class
version throws, and `Trainer` calls it once at construction when `lr_warmup_steps > 0`, so the
failure is **loud and immediate**, not a warm-up that silently does nothing. A negative value
throws too (`TrainerGenericity.WarmupIsRefusedWhereItCannotBeApplied`).

*Trap.* `ScheduleFreeAdamW` has its own `warmup_steps`. It is a different mechanism inside that
optimizer, and `lr_warmup_steps` refuses it (it does not override `set_learning_rate`). Do not set
both expecting them to stack.

**Nested cross-validation.** If you tune hyperparameters using the same
validation split you report your final accuracy on, your reported accuracy
will tend to be optimistic — the model has, in effect, been allowed to "peek"
at the validation data through the hyperparameter search. Nested k-fold
cross-validation [41] avoids this by using two loops: an outer loop that holds
out data purely for the final, unbiased performance estimate, and an inner
loop (run only on the remaining data) that does the actual hyperparameter
search.

### Epoch result

At the end of each epoch (one full pass over the training data), the trainer
reports a summary of what happened:

```cpp
// File: src/core/training/EpochResult.hpp
namespace nn::training
{
struct EpochResult
{
    int epoch = 0;
    float train_loss = 0.0F;
    float val_loss = 0.0F;
    float epoch_ms = 0.0F;

    // SNN energy-efficiency indicators
    // mean_spike_rate: mean firing rate over all SNN neurons in last training batch [0,1].
    // NaN for ANN models (not measured). Target range: [0.05, 0.80] (see SpikeCountLoss).
    float mean_spike_rate = std::numeric_limits<float>::quiet_NaN();

    // sops: estimated Synaptic OPerations for one forward pass.
    // SOPs = Σ_layer(total_spikes × fan_out).  0 when not measured.
    // Compare against ANN FLOP count to quantify energy advantage [26].
    long long sops = 0LL;
};
}
```

`mean_spike_rate` and `sops` only make sense for spiking networks — they stay
`NaN`/`0` for ordinary (non-spiking) models, since there's no meaningful value
to report. **SOPs** ("Synaptic Operations") is the spiking-network equivalent
of a FLOP count: instead of counting floating-point multiply-adds, it counts
how many spikes actually fired, weighted by how many downstream connections
("fan-out") each firing neuron has — comparing this number to an equivalent
ANN's FLOP count is one way to quantify the energy-efficiency argument for
spiking networks (see
[SNN and Surrogate Gradients — Plain](../Concepts/Plain/SNN-and-Surrogate-Gradients.md)
for why fewer/sparser spikes correspond to less energy used):

$$\text{SOPs} = \sum_l \left(\sum_{i,f} s_{i,f}^{(l)}\right) \times \text{fan\_out}^{(l)}$$

where $l$ indexes the layers, the inner sum counts how many spikes occurred in
one batch, and fan_out is how many downstream connections each neuron in that
layer has.

### Trainer

```cpp
// File: src/core/training/Trainer.hpp
template <typename ModelType,
           typename LossType = MSELossImpl<nn::Backend>>
class Trainer
{
public:
    using Sample      = nn::Tensor;
    using SamplePair  = std::pair<nn::Tensor, nn::Tensor>;
    // Optional per-sample preprocessing (encoding, model state reset, etc.)
    using SampleTransform = std::function<nn::Tensor(const nn::Tensor&, std::size_t idx)>;

    explicit Trainer(ModelType& model, const TrainerConfig& cfg);
    explicit Trainer(ModelType& model, const TrainerConfig& cfg, LossType loss);

    // Register a callback (called at train/epoch/batch boundaries)
    void add_callback(std::shared_ptr<ITrainingCallback> cb);

    // Optional per-sample transform applied before model.forward() in both train and val.
    // Use for: encoding, model.reset_state() side effects, data augmentation.
    void set_sample_transform(SampleTransform fn);

    // For autoencoders (input = target)
    auto fit_autoencoder(
        const std::vector<Sample>& train_samples,
        const std::vector<Sample>& val_samples = {}
    ) -> std::vector<EpochResult>;

    // For supervised learning (input != target)
    auto fit_supervised(
        const std::vector<SamplePair>& train_pairs,
        const std::vector<SamplePair>& val_pairs = {}
    ) -> std::vector<EpochResult>;

    const TrainerConfig& config() const;
};
```

`Trainer` is generic over the loss function type (default
`MSELossImpl<nn::Backend>`, i.e. mean-squared error) — any loss type
used in its place must provide three things: a way to tell it what the target
value is (`set_target`), a `forward()` that computes the loss as a single
1×1-tensor number, and a `backward()` that computes the gradient of that loss
with respect to the model's predictions.

**Fixed relative to an earlier version of this training loop:**
1. Gradients are now zeroed **before** the forward pass rather than after —
   zeroing after left a stale gradient sitting around between the backward
   pass and the next batch's zeroing, which is harmless in the common case but
   fragile.
2. Each batch now runs exactly one forward pass, one loss computation, and one
   backward pass (an earlier version computed the forward pass twice per
   batch, wasting half its compute).
3. The loss function is now a template parameter instead of being hard-wired
   to mean-squared error, so callers can plug in e.g. `SpikeCountLoss` for
   spiking-network training.
4. `snn_lr_scale` is now actually wired through to the optimizer via
   `attach_with_scales()` — an earlier version accepted this config value but
   silently never used it.
5. Supervised training's batches are now genuinely batched: all $B$ samples in
   a batch are stacked into one $(B, D)$ matrix and run through a single
   forward/backward pass, rather than looping over $B$ separate $(1, D)$
   single-sample passes. The single-sample-loop version worked correctly, but
   made the GPU do $B$ tiny, inefficient operations instead of one
   appropriately-sized one — a significant slowdown on small feature sets.
6. `EpochResult.mean_spike_rate` is now populated whenever the loss type
   exposes a `last_mean_rate()` method.
7. `Trainer` itself never writes to the console directly anymore — all
   reporting goes through the callback system described below, so callers can
   redirect, filter, or silence it without touching `Trainer`'s code.

### True-batch supervised training (`fit_loop_supervised`)

Concretely, for each mini-batch the trainer stacks all $B$ samples into one
$(B, D)$ input tensor and one $(B, C)$ target tensor, and runs a single
forward+backward pass over the whole batch at once:

```cpp
// Stack batch
Tensor batch_inp = Tensor::zeros(B, in_cols);
Tensor batch_tgt = Tensor::zeros(B, tgt_cols);
for (size_t k = batch_start; k < batch_end; ++k)
{
    batch_inp.setBlock(row, 0, transform(train_inputs[k], k));
    batch_tgt.setBlock(row, 0, train_targets[k]);
}
// Single forward+backward — one GPU kernel per layer, not B
Tensor output  = model_.forward(batch_inp, true);
loss_.set_target(batch_tgt);
Tensor loss_t  = loss_.forward(output, true);
Tensor d_out   = loss_.backward(output);
model_.backward(d_out);
```

This produces mathematically the same result as averaging $B$ individually-
computed gradients (`CrossEntropyLoss` uses a `mean` reduction over
`x.rows()`, i.e. it averages over the batch dimension) — nothing about *what*
is computed changes. The benefit is purely computational: one appropriately-
sized operation per layer instead of $B$ tiny ones, which matters a great deal
on backends where each individual operation carries fixed overhead regardless
of how much data it processes.

Validation batches are handled the same way: every validation sample is
stacked into one $(N_v, D)$ tensor and forwarded once per epoch, rather than
sample-by-sample.

### Time-major batching (`stack_time_major`, added 2026-09-22)

A plain $(B, D)$ stack has nowhere to put time. When each sample is itself a
$(T, F)$ sequence — which is what a spiking encoder produces — the trainer
instead builds the $(T \cdot B, F)$ **time-major** tensor the SNN layers
require, with `out.at(t * B + b, f) = part_b.at(t, f)`:

```
T = 3 steps, B = 2 samples, F features

row 0 | t0, sample 0 |
row 1 | t0, sample 1 |   <- all of step 0 first
row 2 | t1, sample 0 |
row 3 | t1, sample 1 |   <- then all of step 1
row 4 | t2, sample 0 |
row 5 | t2, sample 1 |
```

Row order is *step-major, then batch* — the invariant `LifBPTTImpl` and
`SpikeTimeLossImpl` both assume (see [Time-Steps](../Concepts/Time-Steps.md)).
`stack_time_major` throws if the parts disagree on shape, and degenerates to
the old $(B, F)$ stacking when $T = 1$, so non-sequence training is unaffected.

### Resetting stateful models between batches (`reset_model_state`)

Batches are independent sequences. A stateful model — LIF membrane `v_mem`,
LSTM hidden/cell — must start each one from zero, or the previous batch leaks
into this one.

The trainer now calls `reset_model_state()` before every `model_.forward(...)`
(both training and validation, both the autoencoder and the supervised loop).
It uses the `detail::has_reset_state` trait, so models without `reset_state()`
are unaffected at compile time.

> **Why this was a silent bug, not a loud one.** The trait existed but nothing
> called it. `LifBPTT` re-zeroes `v_mem` only when the input tensor's *shape*
> changes — and the shape is constant across a whole epoch — so the membrane
> carried over from sample to sample for the entire epoch, and training
> completed normally with a plausible loss. Worse, the evaluation paths *did*
> reset explicitly, so the same network was being trained and evaluated under
> two different regimes. Found during the `meeting01` pre-GridUnesp audit; see
> [Meeting01 § The Missing Time Axis](../Experiments/Meeting01.md#the-missing-time-axis-found--fixed-2026-09-22-second-pre-gridunesp-audit).

## Data Flow

```mermaid
sequenceDiagram
    participant Data
    participant Trainer
    participant Model
    participant Loss
    participant Optim

    loop Epoch
        Data->>Trainer: batches
        loop Batch
            Trainer->>Model: forward(batch)
            Model-->>Trainer: output
            Trainer->>Loss: compute(output, target)
            Loss-->>Trainer: loss
            Trainer->>Model: backward(loss)
            Model-->>Trainer: gradients
            Trainer->>Optim: step(params)
            Optim-->>Model: updated weights
        end
        Trainer->>Data: validation batches
    end
```

## Usage Example

```cpp
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "training/ProgressCallback.hpp"
#include "training/EarlyStoppingCallback.hpp"

// Configure training
nn::training::TrainerConfig config{
    .epochs        = 100,
    .learning_rate = 0.001f,
    .batch_size    = 32,
    .grad_clip_norm = 1.0f,
    .snn_lr_scale  = 1.0f  // 1.0 = disabled; 0.1 = SNN biophysical params
};

// Create trainer (MSELoss default; plug in SpikeCountLoss for SNN)
nn::training::Trainer<MyAutoencoder> trainer(*model, config);

// Add callbacks
trainer.add_callback(std::make_shared<nn::training::ProgressCallback>("Train"));
auto stopper = std::make_shared<nn::training::EarlyStoppingCallback>(/*patience=*/20);
trainer.add_callback(stopper);

// Train autoencoder
std::vector<nn::Tensor> train_data = /* load data */;
std::vector<nn::Tensor> val_data   = /* load validation */;

auto history = trainer.fit_autoencoder(train_data, val_data);

// Results
for (const auto& r : history)
{
    // r.epoch, r.train_loss, r.val_loss, r.epoch_ms
    // r.mean_spike_rate (NaN for ANN), r.sops (0 for ANN)
}
```

### Callback interface

A **callback** is a small object that gets notified at fixed points during
training (start/end of training, start/end of each epoch, start/end of each
batch) — this is how progress reporting, early stopping, and metric logging
all hook into the training loop without `Trainer` itself needing to know
anything about progress bars, stopping criteria, or logging formats:

```cpp
// File: include/training/TrainingState.hpp
namespace nn::training {

struct TrainingState {
    int epoch = 0; int total_epochs = 0;
    int batch = 0; int total_batches = 0;
    float batch_progress = 0.0F;   // fraction of current batch processed, e.g. for sub-batch reporting
    float batch_loss = 0.0F;
    const EpochResult* last_epoch_result = nullptr;
};

} // namespace nn::training
```

```cpp
// File: include/training/ITrainingCallback.hpp — #includes TrainingState.hpp
namespace nn::training {

struct ITrainingCallback {
    virtual void on_train_begin(int total_epochs) {}
    virtual void on_train_end(const std::vector<EpochResult>&) {}
    virtual void on_epoch_begin(const TrainingState&) {}
    virtual void on_epoch_end(const TrainingState&, const EpochResult&) {}
    virtual void on_batch_begin(const TrainingState&) {}
    virtual void on_batch_progress(const TrainingState&) {}
    virtual void on_batch_end(const TrainingState&) {}
    virtual bool should_stop() const { return false; }
    virtual ~ITrainingCallback() = default;
};
} // namespace nn::training
```

### Built-in callbacks

**`ProgressCallback`** — wraps the `ProgressManager` described above to draw
a live progress bar:

```cpp
// File: include/training/ProgressCallback.hpp
trainer.add_callback(std::make_shared<nn::training::ProgressCallback>("LSTM run 1/5"));
// Output: LSTM run 1/5  [=====               ] 25% | train_loss: 0.42 | val_loss: 0.38
```

**`EarlyStoppingCallback`** — monitors the validation loss and signals the
trainer to stop once it hasn't improved for a given number of epochs (the
"patience"), which prevents wasting time training a model well past the point
where it's still getting better (or, worse, starting to overfit):

```cpp
// File: include/training/EarlyStoppingCallback.hpp
auto stopper = std::make_shared<nn::training::EarlyStoppingCallback>(
    /*patience=*/20, /*min_delta=*/1e-8f);
trainer.add_callback(stopper);
// After training:
float best = stopper->best_val_loss(); // for RunMetrics reporting
```

**Arming the patience (`arm_below`, third argument).** A network that sits on a plateau
for its first epochs -- an autoencoder at the error of the mean predictor, MSE ≈ 1.0 on
z-scored windows -- has "not improved for 5 epochs" without having started. Plain patience
stops it there and the run reads as "cannot learn". With `arm_below`, epochs only count once
the monitored loss has reached that level (inclusive); a run that never gets there trains every
epoch:

```cpp
// Patience 5, counted only once the loss is 10% below the mean predictor's (L0 = 1.0 here).
auto stopper = std::make_shared<nn::training::EarlyStoppingCallback>(5, 1e-8F, 0.9F * 1.0F);
// losses 1.0 x12, then 0.8, 0.7, 0.75 x5  ->  armed_epoch() == 13, stops after epoch 19
```

| | default (`arm_below` = +inf) | `arm_below` = 0.9 |
|---|---|---|
| 12 epochs at 1.0 | stops at epoch 6 | keeps training |
| reaches 0.8 at 13 | -- | armed at 13; 5 epochs without a new best -> stop at 19 |
| never below 0.9 | stops after 5 flat epochs | trains all epochs |

Tests: `EarlyStoppingCallback.*` in `src/core/training/tests/trainer_gtest.cpp`. meeting01 computes
the mean predictor's loss per training (`Meeting01Stopping.hpp`) -- see
[Meeting01](../Experiments/Meeting01.md#stopping-rule-and-memory-aware-energy-results_format-7-2026-11-03).

**`BestWeightsCallback<Model>`** (`include/training/BestWeightsCallback.hpp`) -- early stopping
ends training *after* the patience epochs, so the last epoch is up to `patience` epochs past the
best one. This callback copies `model.params()` whenever the validation loss reaches a new best
and writes that copy back in `on_train_end` (Goodfellow et al. 2016, Alg. 7.1):

```cpp
trainer.add_callback(std::make_shared<nn::training::BestWeightsCallback<MyModel>>(model));
```

It **throws** instead of guessing: on a NaN validation loss (no validation set -- nothing to
choose by), on a run that ended without any best epoch, and if the parameter count changed
between snapshot and restore. Tests: `BestWeightsCallback.*` in the same file.

`Trainer` checks every callback's `should_stop()` after each epoch ends; if
any callback returns `true`, the training loop exits early and
`on_train_end` still fires normally.

### Sample transform

Some models need a small transformation applied to each sample right before
it's fed into the model — for example, resetting an LSTM's or SNN's internal
state before a new, independent sequence begins, or converting a raw signal
into a spike encoding:

```cpp
// Reset LSTM state and encode each sample before forward pass
trainer.set_sample_transform(
    [&model, &encoding, seed](const nn::Tensor& s, std::size_t idx) -> nn::Tensor {
        model.reset_state();
        return encode_sample(s, encoding, seed + static_cast<uint32_t>(idx));
    });
```

This transform runs in **both** the training and validation loops, immediately
before `model.forward()` is called.

## Common Pitfalls

1. **Learning rate too high or too low.** Too high causes the loss to
   oscillate or diverge; too low makes training take an impractically long
   time to make visible progress.

2. **Gradient clipping.** Usually essential for RNNs/LSTMs, where gradients
   can spike in size unpredictably; can sometimes hurt convolutional networks
   if set too aggressively, since it can suppress genuinely large (and
   useful) gradients.

3. **Batch size.** A larger batch size gives a more stable, less noisy
   gradient estimate, but can also make the model generalise slightly worse
   to new data — there's a real trade-off here, not a universally "correct"
   value.

4. **Skipping validation.** Always evaluate on held-out validation data during
   training — training loss alone cannot tell you whether the model is
   starting to overfit (memorising the training set rather than learning
   generalisable patterns).

## See Also

- [Optimizers](./Optimizers.md) — the pluggable `Optimizer` base class and per-group learning rates (`attach_with_scales`)
- [Layers](./Layers.md) — the model layers being trained
- [Tensor](./Tensor.md) — the underlying data structure
- [Autoencoders](../Concepts/Autoencoders.md) — the model type most commonly trained here
- [K-Fold Cross-Validation](../Concepts/K-Fold-Cross-Validation.md) — the nested cross-validation theory in full
- [Spike Rate Regularization](../Concepts/Spike-Rate-Regularization.md) — what `mean_spike_rate` and SOPs are used for

## References

[1] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. on Learning Representations (ICLR)*, 2015. [Online]. Available: https://arxiv.org/abs/1412.6980

[2] L. Bottou, "Large-scale machine learning with stochastic gradient descent," in *Proc. 19th Int. Conf. Computational Statistics (COMPSTAT)*, 2010, pp. 177–186.

[26] W. Fang et al., "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence," *Science Advances*, vol. 9, no. 40, eadi1480, 2023.

[41] A. Leal et al., "A guide to cross-validation for artificial intelligence in medical imaging," *Radiology: Artificial Intelligence*, 2023. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC10388213/

[74] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 5998–6008. arXiv: [1706.03762](https://arxiv.org/abs/1706.03762)

[84] R. Xiong, Y. Yang, D. He, K. Zheng, S. Zheng, C. Xing, H. Zhang, Y. Lan, L. Wang, and T.-Y. Liu, "On layer normalization in the Transformer architecture," in *Proc. 37th Int. Conf. Machine Learning (ICML)*, PMLR vol. 119, 2020, pp. 10524–10533. [Online]. Available: https://proceedings.mlr.press/v119/xiong20b.html

# Build an Interactive Scientific GUI for Deep Inspection of the Meeting01 and Thesis Pipelines

## 0. Mission

Build a **research-grade interactive scientific visualization GUI** for this repository.

The purpose is NOT merely to make prettier plots.

The application should allow a researcher to **enter the data pipeline at any point, inspect what happened, move backward and forward through transformations, compare Meeting01 and Thesis experiments, animate temporal/spiking/wavelet processes, inspect paraconsistent feature quality, explore latent spaces, and investigate individual samples down to their raw signal representation.**

Think of the application as a:

> **Scientific Data Observatory / Experimental Microscope**

for the two research pipelines.

The GUI must make it possible to go from:

```text
experiment
    ↓
dataset
    ↓
subject / class / recording
    ↓
raw sample
    ↓
window
    ↓
preprocessing
    ↓
normalization
    ↓
wavelet decomposition
    ↓
feature extraction
    ↓
paraconsistent evaluation
    ↓
feature selection
    ↓
SNN / ANN / LSTM / GRU / Transformer
    ↓
latent representation
    ↓
reconstruction
    ↓
classifier / authentication
    ↓
metrics
```

and inspect any intermediate representation.

The user must be able to **zoom in and zoom out conceptually and numerically**.

Do not implement this as a collection of disconnected plots.

Build a coherent application around a shared data model.

---

# 1. FIRST: STUDY THE REPOSITORY

Before modifying code, inspect the repository and `.wiki/` documentation thoroughly.

In particular, understand:

### Meeting01

Study:

* `Experiments/Meeting01.md`
* Meeting01 configuration
* Meeting01 data loading
* FSDD handling
* windowing
* train/validation splitting
* encodings:

  * direct
  * poisson
  * latency
* SNN architectures:

  * dense
  * conv1d
  * recurrent
* SNN-AE
* LSTM-AE
* GRU-AE
* Transformer-AE
* PCA/reference models
* latent representations
* reconstruction metrics
* LOSO/fold structure
* seed/repeat structure
* structured event JSONL
* experiment results
* `config_selected`
* `epoch`
* `epoch_progress`
* `train_end`
* `config_end`
* `fold_begin`
* `fold_end`
* `session_begin`
* `session_end`

The wiki explicitly documents structured Meeting01 events:

```text
session_begin
fold_begin
fold_end
config_begin
epoch
epoch_progress
train_end
config_end
config_selected
session_end
session_error
```

Use these structured sources whenever possible.

Do NOT parse human-readable logs if structured data already exists.

---

# 2. THESIS PIPELINE

Study the thesis experiment documentation, particularly:

* `Experiments/Thesis.md`
* `Experiments/ParaconsistentBaseline.md`
* `Experiments/ParaconsistentGA.md`
* `Experiments/WaveletAE.md`
* `Core/Paraconsistent.md`
* `Core/Wavelet.md`
* `Core/DataLoaders.md`
* SNN documentation
* LFCC documentation
* spike encoding documentation
* relevant demos

The thesis pipeline contains at least these conceptual routes:

```text
Raw EEG
Raw audio
   │
   ├── handcrafted wavelet features
   │
   ├── SNN-AE
   │
   └── ANN-AE
          ↓
feature representation
          ↓
normalization
          ↓
paraconsistent evaluation
          ↓
feature selection
          ↓
authentication/classification
```

Preserve the actual repository terminology and implementation.

Do not invent a simplified pipeline that contradicts the code.

---

# 3. IMPORTANT SCIENTIFIC PRINCIPLE

The GUI is a **visualization and inspection layer**.

It must NOT silently change:

* preprocessing
* normalization
* wavelet parameters
* SNN parameters
* random seeds
* train/validation/test splits
* fold assignments
* feature calculations
* paraconsistent calculations
* model outputs
* metrics
* experimental results

The GUI must consume existing artifacts whenever possible.

If it needs to reproduce a transformation interactively, implement that transformation using the **same core library implementation** used by the experiment.

Do not create a second, subtly different implementation of the scientific pipeline.

---

# 4. TECHNOLOGY ARCHITECTURE

Investigate the repository before choosing the final GUI technology.

Preferred architecture:

```text
                    Scientific GUI
                         │
                 ┌───────┴────────┐
                 │                │
             GUI State        Data Model
                 │                │
          ┌──────┴──────┐        │
          │             │        │
       Qt Widgets     Plotting    │
          │             │         │
       PySide6       PyQtGraph    │
                         │
                    PyVista/VTK
                         │
                 NumPy / SciPy
                         │
              Existing C++/Python
                 scientific core
```

### Recommended roles

#### PySide6

Use for:

* application shell
* menus
* toolbars
* dockable panels
* tree views
* tables
* selectors
* property inspectors
* tabs
* keyboard shortcuts
* application state
* model/view architecture

Qt's `QMainWindow` and `QDockWidget` are particularly appropriate because the GUI should behave like a professional scientific workstation with movable/floating analysis panels.

Use Qt's model/view architecture for large datasets and experiment tables rather than populating thousands of widgets manually.

#### PyQtGraph

Prefer for:

* raw signal plots
* EEG
* audio waveforms
* spectrograms
* wavelet coefficients
* spike rasters
* membrane potentials
* training curves
* interactive time-series exploration
* rapidly changing plots
* synchronized cursors

#### PyVista / VTK

Prefer for:

* 3D latent spaces
* 3D wavelet trees
* 3D feature landscapes
* paraconsistent surfaces
* architecture/network visualization
* point clouds
* volumetric representations
* interactive 3D exploration

PyVista is specifically intended for scientific 3D data and provides NumPy-oriented structures and an application-oriented plotting layer. It can also be embedded into Qt applications.

#### Matplotlib

Do NOT use Matplotlib as the main interactive GUI plotting engine.

Use it for:

* publication-quality figures
* figure export
* specialized static plots
* exact reproduction of paper figures
* PDF/SVG/PNG export

Animation may use Matplotlib where appropriate, but interactive high-frequency exploration should preferentially use PyQtGraph.

#### Existing C++ ImGui/ImPlot

Inspect the existing ImGui/ImPlot demos before replacing functionality.

The repository already has a spike-plotter using:

* ImGui
* ImPlot
* spike raster plots
* membrane potential plots
* precomputed simulation data

Reuse concepts and potentially reusable C++ components if that is technically advantageous.

Do not duplicate existing scientific visualization logic unnecessarily.

---

# 5. APPLICATION CONCEPT

The application should feel like a combination of:

* scientific oscilloscope
* dataset explorer
* experiment debugger
* neural-network microscope
* wavelet analyzer
* paraconsistent analysis workstation

The user should be able to select an object and progressively drill into it.

For example:

```text
Meeting01
  └── FSDD
      └── speaker 17
          └── recording
              └── window #42
                  └── waveform
                      └── encoding
                          └── spike sequence
                              └── SNN layer
                                  └── neuron #12
                                      └── membrane potential
                                          └── spike
```

Similarly:

```text
Thesis
  └── EEG
      └── subject
          └── trial
              └── window
                  └── wavelet packet
                      └── sub-band
                          └── feature
                              └── α / β
                                  └── G1 / G2
                                      └── D_truth
                                          └── D_penalized
```

---

# 6. MAIN WINDOW

Create a professional dockable workspace.

Suggested default layout:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ File  View  Dataset  Experiment  Analysis  Visualization  Export  Help  │
├──────────────────────────────────────────────────────────────────────────┤
│ Dataset / Experiment / Subject / Sample selectors       [Search...]     │
├───────────────┬──────────────────────────────────────┬───────────────────┤
│               │                                      │                   │
│ DATA          │         MAIN VISUALIZATION          │ INSPECTOR         │
│ EXPLORER      │                                      │                   │
│               │                                      │                   │
│ Experiments   │                                      │ Metadata          │
│  ├ Meeting01  │                                      │ Parameters        │
│  └ Thesis     │                                      │ Statistics        │
│               │                                      │ Provenance        │
│ Datasets      │                                      │                   │
│ Subjects      │                                      │                   │
│ Samples       │                                      │                   │
│               │                                      │                   │
├───────────────┴──────────────────────────────────────┴───────────────────┤
│ Timeline / Range / Animation Controls                                  │
├──────────────────────────────────────────────────────────────────────────┤
│ Status | dataset | sample | transform | dimensions | memory | FPS       │
└──────────────────────────────────────────────────────────────────────────┘
```

Everything should be dockable.

Allow the researcher to rearrange the workspace.

Save/restore workspace layouts.

---

# 7. DATA EXPLORER

Create a hierarchical explorer.

Possible hierarchy:

```text
PROJECT
├── Meeting01
│   ├── datasets
│   ├── folds
│   ├── runs
│   ├── models
│   └── results
│
└── Thesis
    ├── EEG
    ├── Audio
    ├── experiments
    ├── folds
    ├── feature strategies
    ├── autoencoders
    └── authentication
```

Selecting an object updates the rest of the application.

Support:

* search
* filtering
* multi-selection
* bookmarks
* recent items
* favorites
* metadata preview

Do not load enormous datasets entirely into RAM merely because they exist.

Use lazy loading.

---

# 8. RAW SIGNAL VIEW

Provide a high-performance signal viewer.

For audio:

* waveform
* amplitude
* RMS
* zero crossings
* spectrogram
* frequency spectrum
* selected window
* sample index
* duration
* sampling frequency

For EEG:

* channel traces
* channel selection
* channel stacking
* synchronized channels
* channel normalization state
* electrode/channel metadata where available
* frequency spectrum
* band-power representation

Features:

* zoom
* pan
* horizontal range selection
* vertical scaling
* logarithmic axes where meaningful
* cursor
* crosshair
* exact sample readout
* coordinate readout
* synchronized cursors across plots

When the user selects a region, all downstream visualizations should optionally update to that region.

---

# 9. "FOLLOW THE DATA" MODE

This is one of the most important features.

Provide a mode where selecting a sample automatically exposes its entire processing chain.

Example:

```text
RAW
 ↓
WINDOW
 ↓
NORMALIZED
 ↓
WAVELET
 ↓
FEATURES
 ↓
PARACONSISTENT
 ↓
LATENT
 ↓
RECONSTRUCTION
 ↓
CLASSIFICATION
```

Each stage should be clickable.

The user should be able to move:

```text
← upstream
downstream →
```

without losing the selected sample.

Display the transformation metadata between stages.

Example:

```text
Wavelet Packet
────────────────────────
Wavelet: Haar
Level: 4
Subbands: 16
Input length: 256
Boundary mode: ...
```

Do not display metadata that is not actually available.

---

# 10. WAVELET LAB

Create a dedicated Wavelet Analysis workspace.

It should support:

### Time domain

```text
original signal
```

### Wavelet decomposition

Show:

```text
                    Root
                     │
             ┌───────┴───────┐
             │               │
            A0              D0
             │               │
       ┌─────┴─────┐   ┌─────┴─────┐
       A1          D1   A2          D2
       ...
```

Make the tree interactive.

Clicking a node highlights the corresponding frequency/sub-band.

Display:

* coefficients
* energy
* frequency interval
* entropy
* ZCR
* Teager-Kaiser
* jitter
* shimmer
* other descriptors actually produced by the repository

For every node show:

```text
Level
Path
Frequency range
Coefficient count
Energy
Relative energy
Selected features
```

---

# 11. 3D WAVELET VISUALIZATION

Provide an optional 3D representation.

Possible representation:

```text
X = time / coefficient index
Y = decomposition level
Z = coefficient magnitude
```

Alternative:

```text
X = frequency
Y = wavelet level
Z = energy
```

Allow:

* rotation
* zoom
* pan
* point/mesh representation
* color mapping
* clipping
* thresholding
* selecting a sub-band
* animation through time

Use PyVista/VTK rather than trying to force a 2D plotting library into a 3D scientific visualization role.

---

# 12. PARACONSISTENT ENGINEERING LAB

This deserves a first-class GUI.

Do not hide the paraconsistent calculation behind a single number.

Expose:

```text
α
β
G1
G2
D_truth
D_penalized
```

and whatever additional quantities the implementation actually calculates.

Create the paraconsistent plane.

At minimum:

```text
                 G2
                  ↑
                  │
                  │
                  │       ● selected feature set
                  │
                  │
                  └────────────────────→ G1
```

Show:

* Truth
* False
* Inconsistency
* Ambiguity

according to the repository's actual convention.

Do not invent labels if the implementation uses a different formulation.

The selected point should be interactive.

Hovering over a point should reveal:

```text
feature strategy
modality
wavelet
category
α
β
G1
G2
D_truth
D_penalized
```

---

# 13. PARACONSISTENT FEATURE LANDSCAPE

Provide a second visualization:

```text
X = feature configuration
Y = D_truth
Z = D_penalized
```

or an appropriate 2D/3D representation based on available data.

Allow filtering by:

* dataset
* modality
* wavelet
* scale
* feature category
* handcrafted / SNN-AE / ANN-AE
* model
* fold
* seed

The user should be able to select a point and immediately inspect the underlying feature vectors.

---

# 14. FEATURE MATRIX INSPECTOR

Create a matrix viewer.

For a selected feature set:

```text
             feature1 feature2 feature3 ...
sample 001
sample 002
sample 003
...
```

Provide:

* heatmap
* row selection
* column selection
* sorting
* normalization toggle
* min/max
* mean
* standard deviation
* class labels
* sample metadata

Clicking a feature should propagate to:

```text
wavelet sub-band
↓
feature extraction
↓
α / β
↓
paraconsistent point
```

when provenance exists.

---

# 15. SNN LAB

Create a dedicated SNN inspection workspace.

The GUI should visualize the actual SNN dynamics.

Support:

### Network topology

```text
Input
 ↓
Linear
 ↓
LIF
 ↓
Linear
 ↓
LIF
 ↓
Latent
```

Render this in 2D and optionally 3D.

### Per-neuron inspection

Selecting a neuron shows:

* input current
* membrane potential
* threshold
* spike times
* reset events
* firing rate
* membrane trajectory

---

# 16. SPIKE RASTER

Provide a high-performance spike raster:

```text
Neuron
  1 |     •       •
  2 |   •     •
  3 |       •
  4 | •   •     •
    +--------------------> time
```

Allow:

* zoom
* pan
* time selection
* neuron selection
* layer selection
* synchronized membrane-potential plot

Selecting a spike should reveal:

```text
time
neuron
layer
membrane potential
threshold
input current
```

---

# 17. ANIMATED SNN SIMULATION

Provide a real animation mode.

The user should be able to press:

```text
▶ Play
⏸ Pause
⏮ Reset
⏩ Step
```

and see the signal propagate through the network.

At time t:

```text
input
 ↓
spikes
 ↓
membrane potential
 ↓
threshold crossing
 ↓
output spike
```

Animate:

* spike propagation
* membrane potential
* neuron activation
* latent activation
* reconstruction

Allow playback speeds:

```text
0.1×
0.25×
0.5×
1×
2×
5×
10×
```

---

# 18. SNN 3D VISUALIZATION

Provide an optional 3D neural activity visualization.

For example:

```text
Layer 1 neurons      Layer 2 neurons      Latent
       ●                   ●                 ●
       ● ────────────────> ● ──────────────> ●
       ●                   ●                 ●
```

Use:

* node position = neuron
* node brightness/size = activity
* edge opacity = weight magnitude
* animation = temporal activity

Do NOT render every edge of a huge network indiscriminately.

Provide:

* layer selection
* neuron filtering
* top-K active neurons
* weight threshold
* activity threshold

so the visualization remains usable.

---

# 19. LATENT SPACE EXPLORER

This should be one of the major GUI screens.

Support:

* 2D PCA
* 2D t-SNE if already available or explicitly enabled
* UMAP only if dependency is justified
* native 3D latent vectors when dimensionality permits
* PCA 3D projection
* class coloring
* subject coloring
* modality coloring
* model coloring
* experiment coloring

Interaction:

```text
click point
    ↓
selected sample
    ↓
raw waveform
    ↓
feature vector
    ↓
wavelet
    ↓
model
    ↓
reconstruction
```

This cross-linking is essential.

The GUI should never treat plots as isolated pictures.

---

# 20. RECONSTRUCTION VIEW

For every autoencoder:

```text
Original
──────────────────────────────

Reconstruction
──────────────────────────────

Residual
──────────────────────────────
```

Display:

* MSE
* MAE
* R²
* correlation
* any other metric actually available

Provide:

```text
overlay
difference
side-by-side
```

modes.

Animate reconstruction progressively across time if meaningful.

---

# 21. MODEL COMPARISON

Create an experiment comparison table.

Possible dimensions:

```text
Dataset
Model
Modality
Encoding
Architecture
v_th
alpha
latent_dim
seed
fold
epochs
training time
MSE
MAE
R²
D_truth
D_penalized
EER
AUC
```

Only display metrics that actually exist for the selected experiment.

Allow sorting/filtering.

Clicking a row opens the complete experiment inspection view.

---

# 22. MEETING01 VS THESIS COMPARISON

Create a dedicated comparison mode.

The purpose is NOT to claim that the experiments are scientifically identical.

Instead, expose structural similarities and differences.

For example:

```text
                         Meeting01             Thesis
────────────────────────────────────────────────────────
Dataset                  FSDD                  EEG / Audio
Windowing
Encoding
Wavelet
Paraconsistent
SNN-AE
ANN-AE
LSTM-AE
GRU-AE
Transformer-AE
Cross-validation
Seeds
Latent dimension
Metrics
```

Clearly distinguish:

```text
same concept
similar implementation
different implementation
not applicable
unknown
```

Never imply equivalence merely because two experiments use SNNs or paraconsistent engineering.

---

# 23. PIPELINE GRAPH

Provide a visual DAG of the current experiment.

Example:

```text
                 ┌─────────────┐
                 │ Raw Signal  │
                 └──────┬──────┘
                        │
             ┌──────────┴──────────┐
             ↓                     ↓
        Wavelet                Autoencoder
             │                     │
             ↓                     ↓
        Features                Latent
             │                     │
             └──────────┬──────────┘
                        ↓
                 Paraconsistent
                        │
                        ↓
                 Feature Ranking
                        │
                        ↓
                  Classifier
                        │
                        ↓
                    Metrics
```

Nodes must be clickable.

Clicking a node opens the appropriate inspector.

---

# 24. ANIMATION SYSTEM

Build a general animation controller rather than implementing unrelated animation code for each plot.

Required controls:

```text
Play
Pause
Stop
Step backward
Step forward
Loop
Speed
Timeline
Current frame
Total frames
```

Animations should work for:

* SNN spikes
* membrane potential
* wavelet decomposition
* signal windows
* reconstruction
* latent trajectories
* training progression where data permits

The animation engine should not recompute expensive transformations on every frame unless necessary.

Cache derived representations.

---

# 25. TIME SYNCHRONIZATION

This is important.

When multiple visualizations represent the same temporal signal, synchronize them.

For example:

```text
Raw signal
      │
      ├── spectrogram
      │
      ├── wavelet
      │
      └── SNN spike train
```

Moving the cursor to:

```text
t = 183 ms
```

should move the cursor in every compatible panel.

Likewise, selecting:

```text
t = 120–220 ms
```

should optionally restrict all downstream plots to that region.

---

# 26. MULTI-RESOLUTION DATA EXPLORATION

Implement an explicit concept of:

```text
Overview
 ↓
Subset
 ↓
Sample
 ↓
Window
 ↓
Time range
 ↓
Feature
 ↓
Coefficient
 ↓
Neuron
 ↓
Timestep
```

The GUI must remain responsive when datasets are large.

Do not render millions of points at full resolution simultaneously.

Use:

* decimation
* level-of-detail rendering
* lazy loading
* caching
* downsampling for overview
* full-resolution rendering only when zoomed sufficiently

Never silently replace analytical values with downsampled values when the user is inspecting exact numerical data.

Clearly distinguish:

```text
DISPLAY-DOWNSAMPLED
```

from:

```text
FULL RESOLUTION
```

---

# 27. DATA PROVENANCE

Every visualization should expose provenance.

For example:

```text
Source
──────
Experiment: Thesis Experiment05
Dataset: EEG
Subject: S07
Trial: 14
Window: 3

Processing
──────────
Wavelet: Haar
Level: 4
Normalization: [0,1]
Feature category: ...
Encoding: latency
Time steps: 16

Model
─────
SNN-AE
v_th: 0.2
latent_dim: 8
seed: 42
```

Where possible provide:

```text
source file
artifact
experiment ID
fold
run
seed
configuration
git commit
```

The Meeting01 event schema already records reproducibility information such as seed, caps, backend and git commit. Surface this.

---

# 28. RAW ARTIFACT INSPECTOR

When the selected object originates from a file:

show:

```text
File
Path
Format
Size
Dataset variable
Shape
dtype
```

For arrays:

```text
shape = (...)
dtype = float32
min
max
mean
std
NaN count
Inf count
```

For matrices:

* row count
* column count
* sparsity if meaningful

For tensors:

* dimensions
* axes
* semantic labels if known

---

# 29. NUMERICAL INSPECTOR

Every plot should provide a way to inspect exact values.

Example:

```text
Cursor: 183
Value: 0.482193
Channel: F3
Time: 183 ms
```

For a feature:

```text
Feature #27
Value: 0.91822
Class: speaker_07
Wavelet: Haar
Band: ...
```

Do not force researchers to infer values from pixels.

---

# 30. EXPORT

Every visualization should support export where meaningful.

Formats:

* PNG
* SVG
* PDF
* CSV
* JSON

For scientific figures:

provide:

```text
Export publication figure
```

with:

* font control
* DPI
* dimensions
* labels
* legend
* transparent background
* vector output

Matplotlib can be used here as a publication backend if advantageous.

---

# 31. EXPERIMENT REPRODUCTION

Selecting an experiment should provide:

```text
[Open experiment]
[Inspect artifacts]
[Show configuration]
[Show provenance]
[Open result files]
```

If safe and already supported by the repository, optionally provide:

```text
[Re-run]
```

BUT:

Do not execute experiments automatically merely because the GUI opened.

Execution must be an explicit user action.

---

# 32. PERFORMANCE ARCHITECTURE

The GUI must never block because a large dataset is being loaded.

Use:

```text
GUI thread
     │
     ├── lightweight state
     │
     └── worker threads/processes
             │
             ├── loading
             ├── preprocessing
             ├── wavelet calculation
             ├── projection
             └── expensive visualization preparation
```

Use caching aggressively.

Possible architecture:

```text
DataRepository
        ↓
DataCache
        ↓
TransformationCache
        ↓
ViewModel
        ↓
Qt Views
```

Avoid recomputing the same wavelet decomposition or latent projection repeatedly.

---

# 33. SCIENTIFIC INTEGRITY RULES

The GUI must distinguish:

### Measured

Data directly loaded from artifacts.

### Computed

Values deterministically recomputed from source data.

### Projected

PCA/t-SNE/etc.

### Display-only

Downsampled/decimated visual representations.

### Estimated

Values explicitly marked as estimates.

### Missing

No available value.

Never substitute:

```text
N/A → 0
```

or invent values for unavailable metrics.

Never call something:

```text
best
optimal
significant
converged
overfitted
generalizing
```

unless the underlying experiment explicitly establishes that condition.

This follows the same scientific-discipline principle as the research TUI.

---

# 34. COLOR SEMANTICS

Use restrained scientific color semantics.

For example:

* raw signal
* transformed signal
* prediction
* reconstruction
* residual
* selected item
* warning
* error

But never make information dependent solely on color.

Use:

* line style
* markers
* labels
* symbols

as additional encodings.

Support dark and light themes.

---

# 35. SEARCH

Provide global search.

Search should find:

* experiment
* dataset
* subject
* recording
* sample
* model
* configuration
* wavelet
* feature
* neuron
* fold
* seed
* result artifact

Example:

```text
Search: "Haar speaker 07"

→ Thesis
   EEG
   speaker 07
   Haar
   Phase 00
   configuration ...
```

---

# 36. BOOKMARKS

Allow researchers to bookmark an interesting state.

A bookmark should store enough information to restore:

```text
dataset
sample
window
experiment
model
fold
seed
visualization
camera position
plot ranges
selected features
selected neuron
```

Example:

```text
Bookmark:
"Interesting SNN latent collapse"
```

Clicking it restores the workspace.

---

# 37. DEBUG / DEVELOPER MODE

Add an optional developer panel.

Show:

```text
GUI FPS
render time
data loading time
cache hit rate
memory usage
active workers
queue size
dataset dimensions
```

Also expose scientific consistency checks:

```text
raw shape
transformed shape
latent shape
reconstruction shape
```

---

# 38. ACCESSIBILITY / FALLBACK

The GUI should remain useful on weaker hardware.

Allow disabling:

* 3D rendering
* animation
* anti-aliasing
* expensive projections
* live updates

Provide a "low performance mode".

---

# 39. 2D/3D VISUALIZATION SELECTION

Do not force every visualization into 3D.

3D should be used when it provides an additional dimension that is scientifically meaningful.

Good candidates:

* latent spaces
* wavelet coefficient landscapes
* neural network topology/activity
* feature landscapes
* volumetric data

Bad candidates:

* ordinary time series
* simple training curves
* tables

Use the simplest representation that preserves the information.

---

# 40. IMPLEMENTATION STRUCTURE

Create a clear separation such as:

```text
gui/
├── app/
│   ├── main.py
│   ├── application.py
│   └── workspace.py
│
├── models/
│   ├── experiment_model.py
│   ├── dataset_model.py
│   ├── sample_model.py
│   ├── feature_model.py
│   └── provenance_model.py
│
├── data/
│   ├── repository.py
│   ├── loaders/
│   ├── cache.py
│   └── artifacts.py
│
├── processing/
│   ├── wavelet.py
│   ├── paraconsistent.py
│   ├── projections.py
│   └── adapters.py
│
├── views/
│   ├── explorer.py
│   ├── signal_view.py
│   ├── wavelet_view.py
│   ├── paraconsistent_view.py
│   ├── snn_view.py
│   ├── latent_view.py
│   ├── reconstruction_view.py
│   ├── comparison_view.py
│   └── pipeline_view.py
│
├── visualization/
│   ├── pyqtgraph/
│   ├── pyvista/
│   └── matplotlib/
│
└── tests/
```

Do not follow this exact structure blindly. Adapt to repository conventions.

---

# 41. DATA ADAPTERS

Do not couple the GUI directly to arbitrary result-file formats.

Create adapters.

Conceptually:

```python
class ExperimentAdapter:
    ...

class Meeting01Adapter(ExperimentAdapter):
    ...

class ThesisAdapter(ExperimentAdapter):
    ...
```

Both should expose a common conceptual interface:

```text
list_datasets()
list_experiments()
list_runs()
list_folds()
list_samples()

load_sample()
load_signal()
load_features()
load_wavelet()
load_latent()
load_metrics()
load_provenance()
```

The GUI should not care whether the underlying artifact came from Meeting01 or Thesis.

---

# 42. CROSS-LINKING API

Every visualization should communicate through a common selection state.

For example:

```python
SelectionState(
    experiment=...,
    dataset=...,
    subject=...,
    recording=...,
    sample=...,
    window=...,
    feature=...,
    wavelet_node=...,
    model=...,
    layer=...,
    neuron=...,
    timestep=...
)
```

Changing one selection should emit an event.

Views subscribe to the event.

This prevents the application from becoming a collection of tightly coupled widgets.

---

# 43. EXAMPLE USER JOURNEYS

The implementation is successful only if workflows such as these are natural.

### Journey A — inspect a thesis sample

```text
Select Thesis
→ EEG
→ subject 07
→ trial 14
→ window 3

See raw EEG

Click Wavelet
→ inspect decomposition

Click sub-band
→ inspect coefficients

Click feature
→ inspect feature vector

Click paraconsistent point
→ inspect α β G1 G2 D_truth D_penalized

Click latent representation
→ inspect SNN-AE latent

Click neuron
→ inspect membrane potential and spikes
```

---

### Journey B — understand SNN encoding

```text
Select sample
→ select SNN-AE
→ select encoding = latency

Play animation

Observe:
normalized value
→ spike timing
→ LIF membrane
→ threshold
→ latent activity
→ reconstruction
```

Then switch:

```text
latency → poisson → direct
```

and compare the resulting representations.

Do not imply causality from visual differences; simply show the observed transformations.

---

### Journey C — investigate a surprising result

```text
Select experiment result
→ unusually low D_penalized

Inspect feature configuration

→ inspect α β
→ inspect G1 G2
→ inspect feature matrix
→ inspect latent activity
→ inspect firing rate
→ inspect reconstruction
→ inspect original signal
```

The application should make this kind of forensic investigation easy.

---

### Journey D — compare Meeting01 and Thesis

```text
Select Meeting01 experiment
Select Thesis experiment

Compare:

pipeline
dataset
windowing
encoding
wavelet
paraconsistent scoring
model
latent representation
metrics
```

Then drill into corresponding samples/representations where a meaningful mapping exists.

---

# 44. WAVELET + SNN + PARACONSISTENT "TRIANGLE" VIEW

Create one particularly interesting experimental view.

Three synchronized panels:

```text
┌─────────────────┬─────────────────┬──────────────────┐
│ WAVELET         │ SNN             │ PARACONSISTENT   │
│                 │                 │                  │
│ coefficients    │ spikes          │ α β              │
│ energy          │ membrane        │ G1 G2            │
│ bands           │ latent          │ D_truth          │
│                 │ reconstruction  │ D_penalized      │
└─────────────────┴─────────────────┴──────────────────┘
```

Selecting a feature/sample in any panel highlights the corresponding object in the other panels.

This should become one of the application's signature views.

---

# 45. EXPERIMENT TIMELINE

For experiments with structured event data, create a timeline.

For Meeting01, use the existing event stream.

Display:

```text
session
 │
 ├ fold 0
 │   ├ config 1
 │   │   ├ epoch 1
 │   │   ├ epoch 2
 │   │   └ ...
 │   ├ config 2
 │   └ ...
 │
 ├ fold 1
 ...
```

Allow the user to select an event and jump directly to the corresponding experiment/configuration.

---

# 46. TRAINING VISUALIZATION

For selected training runs show:

* train loss
* validation loss
* learning rate
* epoch duration
* best epoch
* convergence trajectory
* reconstruction examples

If intra-epoch events exist, expose them.

The Meeting01 `epoch_progress` event should be visualized where available.

---

# 47. RESULT RANKING

Provide a ranking panel.

Allow ranking by available metrics:

```text
validation loss
test loss
MSE
MAE
R²
D_truth
D_penalized
EER
AUC
training time
inference cost
parameter count
```

Never mix incomparable metrics without explicitly identifying the metric and direction.

---

# 48. PARACONSISTENT GA

If `paraconsistentGA` results exist, provide a dedicated NSGA-II view.

Display:

```text
population
generation
individual
genome
fitness
feasibility
latency
latent activity
D_penalized
inference cost
```

Visualize:

```text
D_penalized
      ↑
      │       ○
      │    ○
      │ ○     ● ●
      │   ●
      └────────────────→ inference cost
```

Highlight the feasible Pareto front.

Allow clicking an individual to inspect:

```text
genome
→ architecture
→ latent
→ paraconsistent score
→ reconstruction
```

Respect the repository's explicit warning that estimated latency is **UNCALIBRATED** where applicable.

---

# 49. TESTING

Add tests for:

### Data model

* dataset discovery
* sample selection
* provenance
* shape propagation

### Visualization

* empty dataset
* one sample
* missing metric
* missing artifact
* large dataset
* NaN
* Inf

### Synchronization

* selection propagation
* cursor synchronization
* time-range synchronization

### Scientific correctness

Compare GUI-derived values against known reference values for:

* normalization
* wavelet coefficients
* α
* β
* G1
* G2
* D_truth
* D_penalized
* spike counts
* membrane potential

The GUI must never silently alter numerical results.

---

# 50. PERFORMANCE TESTING

Test with:

* tiny dataset
* normal dataset
* large dataset
* many experiments
* thousands of samples
* long signals
* large latent spaces

Measure:

```text
startup
dataset discovery
sample load
plot update
3D update
animation FPS
memory
cache effectiveness
```

---

# 51. DOCUMENTATION

Create documentation explaining:

* how to start the GUI
* architecture
* supported experiments
* data adapters
* visualization architecture
* caching
* adding new experiments
* adding new visualizations
* scientific provenance
* limitations

Include screenshots/examples if practical.

---

# 52. DO NOT OVERENGINEER THE FIRST VERSION

Implement in stages.

## Phase 1 — Foundation

* PySide6 application shell
* dockable workspace
* experiment explorer
* dataset/sample selector
* provenance inspector
* raw signal viewer

## Phase 2 — Scientific pipeline

* wavelet viewer
* feature matrix
* paraconsistent plane
* metrics

## Phase 3 — Neural visualization

* SNN raster
* membrane potential
* latent space
* reconstruction
* network topology

## Phase 4 — 3D

* PyVista integration
* latent 3D
* wavelet 3D
* network activity 3D

## Phase 5 — Animation

* common timeline
* synchronized playback
* SNN animation
* wavelet animation
* reconstruction animation

## Phase 6 — Cross-experiment analysis

* Meeting01 adapter
* Thesis adapter
* comparison view
* pipeline graph
* experiment timeline

## Phase 7 — Research polish

* bookmarks
* export
* publication figures
* performance mode
* provenance
* tests
* documentation

Do not implement all features as one enormous untestable first commit.

---

# 53. DEPENDENCY DECISION

Before adding dependencies, inspect what is already installed/used.

Preferred candidates:

```text
PySide6
PyQtGraph
PyVista
VTK
NumPy
SciPy
Pandas
Matplotlib
```

Potentially:

```text
scikit-learn
```

only where projections/analysis are actually required.

Do not add Plotly/Dash merely because it can visualize something.

A browser-based Dash application is an alternative architecture, but for this repository I currently prefer a native Qt application because the requirements emphasize deep hierarchical inspection, docking, synchronized scientific views, desktop interaction, and potentially local access to existing artifacts. Dash/Plotly is nevertheless a reasonable future web-viewer/export option; Plotly provides interactive WebGL-capable graphing, while Dash VTK provides VTK-based 3D components.

---

# 54. IMPORTANT: INVESTIGATE BEFORE DECIDING

Do not blindly implement the proposed technology stack.

First answer:

1. What visualization infrastructure already exists?
2. Which data formats already exist?
3. Which computations already exist in C++?
4. Which computations already exist in Python?
5. Which results are persisted?
6. Which results can be regenerated?
7. Which parts should remain in C++?
8. Which parts are better exposed through Python?
9. Is there already an ImGui/ImPlot application architecture that should be extended?
10. What is the cheapest architecture that provides the required interaction?

Then choose the architecture.

---

# 55. MOST IMPORTANT DESIGN RULE

The GUI should answer:

> **"Why did this number become this number?"**

not merely:

> **"What is this number?"**

For example:

```text
D_penalized = 0.158
```

should allow the user to navigate toward:

```text
D_penalized
    ↓
D_truth
    ↓
G1 / G2
    ↓
α / β
    ↓
feature distributions
    ↓
feature extraction
    ↓
wavelet coefficients
    ↓
raw signal
```

Similarly:

```text
SNN latent feature
    ↓
latent activation
    ↓
spike train
    ↓
membrane potential
    ↓
input encoding
    ↓
normalized input
    ↓
raw signal
```

That is the core concept of the application.

---

# 56. FINAL ACCEPTANCE CRITERIA

The project is successful when a researcher can:

* browse both Meeting01 and Thesis experiments;
* select a dataset;
* select a subject/recording/sample;
* inspect the raw signal;
* inspect preprocessing;
* inspect wavelet decomposition;
* inspect individual wavelet sub-bands;
* inspect extracted features;
* inspect paraconsistent α/β/G1/G2;
* inspect D_truth and D_penalized;
* inspect SNN encoding;
* inspect spikes;
* inspect membrane potentials;
* inspect latent representations;
* inspect reconstruction;
* inspect experiment metrics;
* compare experiments;
* animate temporal transformations;
* rotate and inspect 3D representations;
* trace a result back to its source;
* trace a raw sample forward through the pipeline;
* export a scientific figure;
* preserve numerical fidelity;
* remain responsive on large datasets.

The GUI should feel less like a "dashboard" and more like a **microscope for the research pipeline**.

Do not merely add more plots.

Build an environment in which the researcher can **interrogate the experiment**.

# ResNet Classifier Demo (resnet_demo)

Demonstrates a residual MLP (ResNet-style skip-connection network) trained on raw audio features from a single MAT subject file. The demo validates that the custom C++ `ResidualBlock`, `CrossEntropyLoss`, and `Adam` implementations compose correctly and converge on a small classification problem.

## Algorithm

### Data loading

Reads variable `"Audio"` from `S02_Audio.mat` (matioCpp). Rows are samples; the last column is treated as an integer class label. Feature matrix $\mathbf{X} \in \mathbb{R}^{N \times F}$ and one-hot label matrix $\mathbf{Y} \in \mathbb{R}^{N \times C}$ are built, where $C = \max(\text{labels}) + 1$.

### Forward pass

$$\mathbf{h}_0 = \text{ReLU}(\mathbf{W}_0 \mathbf{x} + \mathbf{b}_0), \quad \mathbf{h}_0 \in \mathbb{R}^{64}$$

Two residual blocks:

$$\mathbf{h}_{l+1} = \text{ReLU}(\mathbf{W}_{l,2}\,\text{ReLU}(\mathbf{W}_{l,1} \mathbf{h}_l + \mathbf{b}_{l,1}) + \mathbf{b}_{l,2}) + \mathbf{h}_l$$

Output logits (no activation):

$$\hat{\mathbf{y}} = \mathbf{W}_\text{out} \mathbf{h}_2 + \mathbf{b}_\text{out}, \quad \hat{\mathbf{y}} \in \mathbb{R}^{C}$$

### Loss

Categorical cross-entropy with softmax:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{c=1}^{C} y_{ic} \log \frac{e^{\hat{y}_{ic}}}{\sum_{c'} e^{\hat{y}_{ic'}}}$$

### Optimiser

Adam with $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.

## Architecture

```
Input x ∈ R^F
    │
Linear(F → 64)  [Kaiming init]
    │
   ReLU
    │
ResidualBlock(64)
    ├─ Linear(64→64) → ReLU → Linear(64→64) → ReLU
    └─ skip: identity shortcut
    │
ResidualBlock(64)
    ├─ Linear(64→64) → ReLU → Linear(64→64) → ReLU
    └─ skip: identity shortcut
    │
Linear(64 → C)
    │
Output logits ∈ R^C  →  CrossEntropyLoss
```

Training loop: mini-batch SGD, `batch_size=16`, `epochs=1` (quick verification run).

## Theory & State of the Art

Residual connections were introduced by He et al. (2016) to address gradient vanishing in deep networks. The shortcut $\mathbf{h}_{l+1} = F(\mathbf{h}_l) + \mathbf{h}_l$ guarantees a gradient path of unit magnitude regardless of depth, enabling training of networks with hundreds of layers.

For dense (fully-connected) residual blocks — sometimes called ResNet-FC or dense residual networks — the same principle applies to MLPs. In audio feature classification, such architectures achieve competitive results on shallow feature sets (MFCC/LFCC) where kernel-based convolutions offer no structural advantage (Ravanelli & Bengio, 2018).

Kaiming (He) initialisation (He et al., 2015) sets $\sigma_w = \sqrt{2/\text{fan\_in}}$ to maintain variance through ReLU nonlinearities, replacing the Xavier scheme that is optimal for linear or tanh activations.

## How to Use (HOWTO)

### Build

```bash
cd software/nn
cmake --preset=max-performance
cmake --build out/build/max-performance --target resnet_demo -j$(nproc)
```

### Run

```bash
./out/build/max-performance/src/demos/cppDemos/resnet_classifier_demo/resnet_demo
```

Expects `S02_Audio.mat` at a hard-coded path relative to the working directory. Place the file there or adjust the path constant in `resnet_demo.cpp`.

### Expected Output

Per-epoch training loss printed to stdout:

```
Epoch 1/1  loss: <float>
```

The demo does not save weights or compute test accuracy; it is a forward-backward smoke test.

## Dependencies

| Library | Purpose |
|---|---|
| `matioCpp` | Load `.mat` subject file |
| `xtensor`, `xtensor-blas` | Tensor arithmetic |
| `tensor` (project) | `nn::Tensor` type and operations |
| `data_loaders` (project) | Data loading utilities |
| `util` (project) | Shared utilities |

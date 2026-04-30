# LSTM and Backpropagation Through Time (BPTT)

## Theoretical Background

### The Vanishing Gradient Problem

Standard RNNs suffer from vanishing/exploding gradients when learning long sequences [2]:

$$\frac{\partial h_t}{\partial h_{t-\tau}} = \prod_{i=t-\tau}^{t} \frac{\partial h_i}{\partial h_{i-1}}$$

When $|\frac{\partial h_i}{\partial h_{i-1}}| < 1$, gradients decay exponentially; when $> 1$, they explode.

### LSTM Gates

LSTM controls information flow via four gates, verified against [1, 3]:

| Gate | Symbol | Activation | Role |
|------|--------|-----------|------|
| Input | $i_t$ | sigmoid | how much new info to write |
| Forget | $f_t$ | sigmoid | how much of $C_{t-1}$ to keep |
| Output | $o_t$ | sigmoid | what part of cell state to expose |
| Cell candidate | $g_t$ | tanh | new content to potentially write |

### Forward Pass

Weight matrices are stacked as $[i \mid f \mid o \mid g]$ (gate ordering per [3]):

$$\text{pre}_t = x_t W^T + H_{t-1} U^T + b^T \quad \in \mathbb{R}^{1 \times 4H}$$

Gates (each slice of width $H$):

$$i_t = \sigma(\text{pre}_t[0{:}H]), \quad
f_t = \sigma(\text{pre}_t[H{:}2H]), \quad
o_t = \sigma(\text{pre}_t[2H{:}3H]), \quad
g_t = \tanh(\text{pre}_t[3H{:}4H])$$

Cell and hidden state:

$$C_t = f_t \odot C_{t-1} + i_t \odot g_t$$
$$H_t = o_t \odot \tanh(C_t)$$

The cell update contains **no weight matrix on the recurrent path** ($C_t \leftarrow f_t \odot C_{t-1}$) — the "constant error carousel" that prevents vanishing gradients [1].

Forget gate bias is initialised to 1 to discourage forgetting at the start of training [4].

### Backward Pass (BPTT)

Gradients flow backward from $t = T$ to $t = 1$. At each step, let $\delta h_t$ denote
the total gradient at $H_t$ (from loss + recurrent path):

$$\delta h_t = \frac{\partial L}{\partial H_t}\bigg|_\text{output} + \delta h_\text{next}$$

Output gate and cell state:

$$\delta o_t = \delta h_t \odot \tanh(C_t)$$
$$\delta C_t = \delta h_t \odot o_t \odot (1 - \tanh^2(C_t)) + \delta C_\text{next}$$

Gate gradients:

$$\delta i_t = \delta C_t \odot g_t, \quad
\delta f_t = \delta C_t \odot C_{t-1}, \quad
\delta g_t = \delta C_t \odot i_t$$

Cell state gradient propagated to previous step:

$$\delta C_\text{next} \leftarrow \delta C_t \odot f_t$$

Pre-activation gradients (applying activation derivatives):

$$\delta\text{pre}_t = [\,
  \delta i_t \odot i_t(1-i_t) \;\mid\;
  \delta f_t \odot f_t(1-f_t) \;\mid\;
  \delta o_t \odot o_t(1-o_t) \;\mid\;
  \delta g_t \odot (1-g_t^2)
\,]$$

Weight, bias, and input gradients accumulated over $T$ steps:

$$\frac{\partial L}{\partial W} = \sum_t \delta\text{pre}_t^T x_t, \quad
\frac{\partial L}{\partial U} = \sum_t \delta\text{pre}_t^T H_{t-1}, \quad
\frac{\partial L}{\partial b} = \sum_t \delta\text{pre}_t^T$$

$$\frac{\partial L}{\partial x_t} = \delta\text{pre}_t \, W, \quad
\delta h_\text{next} \leftarrow \delta\text{pre}_t \, U$$

## Implementation: `LSTMLayerImpl<Backend>`

**File:** `include/nn/layers/lstm/LSTMLayer.hpp`

### Parameters

| Tensor | Shape | Description |
|--------|-------|-------------|
| `W_` | $(4H, D)$ | Input-to-hidden (gates stacked $[i\|f\|o\|g]$) |
| `U_` | $(4H, H)$ | Hidden-to-hidden |
| `b_` | $(4H, 1)$ | Bias; forget-gate slice initialised to 1 |

### Shape Contract

| Call | Input | Output |
|------|-------|--------|
| `forward(T×D)` | single sequence | `T×H` — persists `h0_`, `c0_` |
| `forward(B×T×D)` | batch | `B×T×H` — each sample starts from zero state |
| `backward(T×H)` | 2-D grad | `T×D` via `_bptt_apply` |
| `backward(B×T×H)` | 3-D grad | `B×T×D` via `backward_3d` |

### `LSTMStepCache` — BPTT cache

Only fields required by `_bptt_pure` are stored (verified against [1, 3]).
$C_t$ and $H_t$ are **not** cached — they are not needed in the backward pass,
saving $2 \times H$ floats per timestep per sequence.

```
x      — x_t
h_prev — H_{t-1}
c_prev — C_{t-1}
i      — σ(pre_i)
f      — σ(pre_f)
o      — σ(pre_o)
g      — tanh(pre_g)
tanh_c — tanh(C_t)     ← used for dL/do and dL/dC
```

### Core Private Methods

| Method | Purpose |
|--------|---------|
| `_run_sequence(seq, h, c, ...)` | shared gate loop for 2-D and 3-D forward |
| `_bptt_pure(cache, dL_dh)` | pure BPTT, returns $(dW, dU, db, dx)$ without writing members |
| `_bptt_apply(cache, dL_dh)` | calls `_bptt_pure`, writes to `dW_/dU_/db_` and sets grads |
| `forward_2d` | dispatches for `(T,D)` input; persists state |
| `forward_3d` | dispatches for `(B,T,D)` input; independent states per sample |
| `backward_3d` | accumulates gradients across batch dimension |

### Usage

```cpp
#include "nn/layers/lstm/LSTMLayer.hpp"

nn::models::lstm::LSTMLayer layer(/*input_size=*/64, /*hidden_size=*/128);

// Single sequence
layer.reset_state();
nn::Tensor out = layer.forward(seq_2d, /*requires_grad=*/true);  // (T,H)
nn::Tensor dx  = layer.backward(grad_out);                        // (T,D)

// Batch (B independent sequences)
nn::Tensor out3 = layer.forward(seq_3d, true);   // (B,T,D) → (B,T,H)
nn::Tensor dx3  = layer.backward(grad_out3);      // (B,T,H) → (B,T,D)

// Optimise
Adam opt;
opt.attach(layer.params());
opt.step();
layer.reset_state();
```

## Common Pitfalls

1. **Call `reset_state()` between independent sequences.** Hidden and cell state persist across `forward()` calls on the 2-D path.
2. **Gradient clipping** — essential for long sequences; clip $\|\nabla W\|$ to $[-\theta, \theta]$ [2].
3. **Forget-gate bias = 1** — already initialised; do not zero-initialise the full bias vector.
4. **Rate of learning** — biophysical SNN params need ~10× lower lr than weights; use `Adam::attach_with_scales()`.

## References

[1] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997. doi: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)

[2] P. J. Werbos, "Backpropagation through time: What it does and how to do it," *Proc. IEEE*, vol. 78, no. 10, pp. 1550–1560, Oct. 1990. doi: [10.1109/5.58337](https://doi.org/10.1109/5.58337)

[3] K. Greff, R. K. Srivastava, J. Koutník, B. R. Steunebrink, and J. Schmidhuber, "LSTM: A search space odyssey," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 28, no. 10, pp. 2222–2232, Oct. 2017. doi: [10.1109/TNNLS.2016.2582924](https://doi.org/10.1109/TNNLS.2016.2582924). arXiv: [1503.04069](https://arxiv.org/abs/1503.04069)

[4] R. Jozefowicz, W. Zaremba, and I. Sutskever, "An empirical evaluation of recurrent network architectures," in *Proc. 32nd Int. Conf. Mach. Learn. (ICML)*, 2015, pp. 2342–2350. [Online]. Available: http://proceedings.mlr.press/v37/jozefowicz15.pdf

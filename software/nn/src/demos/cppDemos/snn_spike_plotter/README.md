# SNN Spike Plotter (plotSpikingNetwork)

Real-time interactive visualisation of a two-neuron LIF (Leaky Integrate-and-Fire) chain driven by Poisson spike input. Renders a spike raster and a continuous membrane-potential trace side by side using ImGui + ImPlot, allowing visual verification of neuron dynamics, threshold crossing, and reset behaviour without a training loop.

## Algorithm

### Input generation

Synthetic spike input is generated via `generate_autoencoder_spike_data(1, 1, n_steps=200, max_rate=0.5, dt=1.0)`. For each time step $t$:

$$s_\text{in}[t] \sim \text{Bernoulli}(r_{\max}), \quad r_{\max} = 0.5$$

### LIF dynamics (single-step)

Each neuron uses `Lif` (single-step variant, not BPTT). Membrane potential update:

$$V[t+1] = \beta\, V[t] + (1-\beta)\, R\, I[t] - V_{\text{th}}\, S[t]$$

$$\beta = e^{-\Delta t / (R \cdot C)}, \quad \Delta t = 1.0,\; R = 3.0,\; C = 2.0,\; V_{\text{th}} = 1.0$$

$$S[t] = \mathbf{1}[V[t] \geq V_{\text{th}}]$$

Two neurons chained: `hidden_neuron` receives the Poisson input; `output_neuron` receives `hidden_neuron`'s spike output.

### Visualisation

ImGui frame loop:
- `ImGui::Begin("Neuron Output")`
- **Spike raster panel** (ImPlot scatter): time vs spike identity (0 = hidden, 1 = output)
- **Membrane potential panel** (ImPlot line): $V_\text{hidden}[t]$ over the full $n_\text{steps} = 200$ window

## Architecture

```
Poisson source  (max_rate=0.5, 200 steps)
       │  s_in[t] ∈ {0,1}
       ▼
  Lif hidden_neuron  (R=3, C=2, V_th=1.0)
       │  S_h[t], V_h[t]
       ▼
  Lif output_neuron  (same params)
       │  S_o[t], V_o[t]
       ▼
  ImGui/ImPlot window
    ├─ scatter: spike times for both neurons
    └─ line:    V_h[t] over time
```

No training, no backprop. Pure forward simulation of 200 steps replayed each render frame.

## Theory & State of the Art

The Leaky Integrate-and-Fire model is the canonical point-neuron model in computational neuroscience (Lapicque, 1907; Gerstner & Kistler, 2002). The LIF combines a resistor-capacitor (RC) circuit analog with a hard threshold and reset:

$$C \frac{dV}{dt} = -\frac{V}{R} + I(t), \quad V \leftarrow 0 \text{ if } V \geq V_{\text{th}}$$

Discretised with step $\Delta t$ this yields the update rule above. The decay factor $\beta = e^{-\Delta t/(RC)}$ determines the membrane time constant $\tau_m = RC$.

For $R = 3.0$, $C = 2.0$: $\tau_m = 6.0\,\text{ms}$ (with $\Delta t = 1\,\text{ms}$); $\beta \approx 0.846$. With $\Delta t = 1.0$ (as set in this demo) the time axis is dimensionless and serves only as a visualisation aid.

The Poisson spike model (rate coding) is the simplest biologically plausible encoding scheme (Dayan & Abbott, 2001). Real neurons exhibit near-Poisson variability in cortical areas, making this a reasonable first-order approximation for probing SNN layer dynamics.

ImGui (Dear ImGui, Cornut, 2014) combined with ImPlot (Epstein, 2020) provides a lightweight, immediate-mode GUI suitable for real-time scientific visualisation without a full plotting framework.

## How to Use (HOWTO)

### Build

```bash
cd software/nn
cmake --preset=max-performance
cmake --build out/build/max-performance --target plotSpikingNetwork -j$(nproc)
```

Requires a display (X11/Wayland) and an OpenGL-capable GPU or software renderer.

### Run

```bash
./out/build/max-performance/src/demos/cppDemos/snn_spike_plotter/plotSpikingNetwork
```

No arguments. The simulation parameters are compile-time constants in `plotSpikingNetwork.cpp`.

### Expected Output

An interactive GUI window "Neuron Output" opens with two panels:

1. **Spike raster** — scatter plot showing spike events for the hidden and output neurons over 200 time steps.
2. **Membrane potential** — continuous line plot of $V_\text{hidden}[t]$, showing RC charging and reset events.

Close the window to exit.

## Dependencies

| Library | Purpose |
|---|---|
| `OpenGL` | GPU-accelerated rendering backend |
| `GLFW` | Window creation and input handling |
| `imgui` | Immediate-mode GUI framework |
| `implot` | ImGui extension for scientific plots |
| `cnpy` | NumPy `.npy` I/O (linked; not used in main flow) |
| `xtensor`, `xtensor-blas` | Tensor arithmetic |
| `tensor` (project) | `nn::Tensor` and `Lif` layer |
| `util` (project) | Shared utilities |

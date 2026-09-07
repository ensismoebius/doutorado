# SNN Spike Plotter

Real-time interactive visualisation of a two-neuron LIF chain driven by Poisson spike input. Renders a spike raster and continuous membrane-potential trace side by side using ImGui + ImPlot, allowing visual verification of neuron dynamics, threshold crossing, and reset behaviour without a training loop.

---

## Theoretical Background

The Leaky Integrate-and-Fire model is the canonical point-neuron model in computational neuroscience [Lapicque, 1907; Gerstner & Kistler, 2002]:

$$C \frac{dV}{dt} = -\frac{V}{R} + I(t), \qquad V \leftarrow 0 \text{ if } V \geq V_\text{th}$$

Discretised with step $\Delta t$: $V[t] = \beta V[t-1] + R \cdot I[t]$, $\beta = e^{-\Delta t/(RC)}$.

For $R = 3$, $C = 2$: $\tau_m = 6$, $\beta \approx 0.846$ (slow decay, integrates well).

The Poisson spike model [Dayan & Abbott, 2001] generates $s[t] \sim \text{Bernoulli}(r_{\max})$. With $r_{\max} = 0.5$, roughly half of time steps carry a spike. This is the simplest rate-coded encoding scheme.

---

## How It Is Implemented Here

**Source:** `src/demos/cppDemos/snn_spike_plotter/plotSpikingNetwork.cpp`

```cpp
// plotSpikingNetwork.cpp (structure)
// 1. Generate spike input via generate_autoencoder_spike_data(1, 1, n_steps=200, max_rate=0.5, dt=1.0)
// 2. Two Lif neurons (single-step): dt=1.0, R=3, C=2, V_th=1.0, reset_zero=true
// 3. run_simulation(): ONCE, before the render loop — for t in 0..200:
//      hidden_out = hidden_neuron.forward(spike_inputs[t])
//      output_out = output_neuron.forward(hidden_out)
//      record spikes + v_mem into SimulationResult sim
// 4. ImGuiApp::run() render loop (every frame, no re-simulation):
//      ImGui::Begin("Neuron Output")
//      draw_spike_raster_plot(sim)   // 3x ImPlot::PlotScatter: Input/Hidden/Output
//      draw_membrane_potential_plot(sim) // 2x ImPlot::PlotLine: Hidden/Output V_mem
```

No training; the 200-step forward simulation runs exactly once before the window opens —
the render loop only re-draws the same precomputed `SimulationResult` every frame.

---

## Data Flow

```mermaid
flowchart TD
    A["Poisson source\n max_rate=0.5, 200 steps\n s_in[t] ∈ {0,1}"] --> B["Lif hidden_neuron\n dt=1, R=3, C=2, V_th=1.0"]
    B --> C["Spikes S_h[t] + membrane V_h[t]"]
    C --> D["Lif output_neuron\n same params"]
    D --> E["Spikes S_o[t] + membrane V_o[t]"]
    C --> S["SimulationResult sim\n (computed once, before render loop)"]
    E --> S
    S --> F["Every render frame:\n draw_membrane_potential_plot\n PlotLine V_h, V_o"]
    S --> G["Every render frame:\n draw_spike_raster_plot\n PlotScatter Input/Hidden/Output"]
    F --> H["ImGui window\n 'Neuron Output'"]
    G --> H
```

---

## How to Build and Run

```bash
cd /home/ensismoebius/Repos/doutorado/software/nn
cmake --preset=max-performance
cmake --build out/build/max-performance --target plotSpikingNetwork -j$(nproc)
./out/build/max-performance/src/demos/cppDemos/snn_spike_plotter/plotSpikingNetwork
```

Requires a display (X11/Wayland) and OpenGL. On headless servers, use a virtual framebuffer (`Xvfb`).

**Expected output:** interactive GUI window with two panels — spike raster and membrane potential.

---

## Test Suite

The demo has its own gtest target (`LifTest` fixture — forward shape/binariness, threshold
firing, `reset_state()`, membrane decay, backward gradient shape/finiteness):

```bash
cmake --build out/build/max-performance --target snn_spike_plotter_gtest -j$(nproc)
ctest --test-dir out/build/max-performance -R LifTest --output-on-failure
```

---

## Common Pitfalls

1. **No display available**: ImGui requires an OpenGL context. On CI or SSH servers without `DISPLAY`, the binary will crash at GLFW init. Use `Xvfb :99 &` and `DISPLAY=:99` as a workaround.
2. **Single-step vs BPTT**: this demo uses `LifImpl` (single-step), not `LifBPTTImpl`. The single-step variant does not accept a `(T*B, F)` input — it processes one time step per call. Do not substitute `LifBPTTImpl` here without restructuring the loop.
3. **Simulation runs once, not per frame**: `run_simulation()` executes all 200 steps before `window.run()` starts; the render callback only redraws the precomputed `SimulationResult`. To make the simulation live/continuous instead of a fixed replay, step the neurons and append to `sim` from inside the render callback.

---

## See Also

- [Concepts/Membrane-Dynamics](../Concepts/Membrane-Dynamics.md) — LIF RC circuit theory
- [Concepts/SNN-and-Surrogate-Gradients](../Concepts/SNN-and-Surrogate-Gradients.md) — training spiking networks
- [Core/Layers](../Core/Layers.md) — `LifImpl` and `LifBPTTImpl` comparison

---

## References

[1] L. Lapicque, "Recherches quantitatives sur l'excitation électrique des nerfs traitée comme une polarisation," *J. Physiol. Pathol. Gen.*, vol. 9, pp. 620–635, 1907.

[2] W. Gerstner and W. M. Kistler, *Spiking Neuron Models*. Cambridge, UK: Cambridge University Press, 2002.

[3] P. Dayan and L. F. Abbott, *Theoretical Neuroscience*. Cambridge, MA: MIT Press, 2001.

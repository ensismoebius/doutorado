# LeakyIntegrator Readout Layer (Standalone Module)

The `LeakyIntegrator` is a non-spiking variant of the Leaky Integrate-and-Fire (LIF) neuron. It acts as a continuous-valued readout layer, often used at the final output of an SNN to transform spike trains back into a continuous signal (e.g., for regression tasks or reconstruction).

Conceptually:
- **LIF**: $V_t = \beta V_{t-1} + I_t$. If $V_t > V_{th} \rightarrow$ Spike & Reset.
- **LeakyIntegrator**: $V_t = \beta V_{t-1} + I_t$. Output is $V_t$ directly (no threshold, no reset).

This decoupling is vital for autoencoders trained with MSE loss, as discrete spikes cannot easily match a target continuous value (like 1.0) without quantization error. The LeakyIntegrator acts as a low-pass filter (smoother) on the incoming spike train.

## Role & Usage
- **As a Readout Layer (Recommended)**: Use it as the final layer of your decoder.
- **Diagnostics**: Can replace hidden LIF layers to debug gradient flow (temporarily turning the SNN into a standard RNN).
- **Not an Input Encoder**: It does not convert analog values to spikes; it does the reverse (spikes to analog potential).

## Usage Example
```cpp
#include "nn/layers/LeakyIntegrator.hpp"

// ...
auto readout = std::make_shared<LeakyIntegrator>(
    time_step, // dt
    resist,    // R
    capct      // C
);
// Output is continuous membrane potential, trainable via BPTT.
```

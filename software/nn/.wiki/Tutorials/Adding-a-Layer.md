# Tutorial 2 — Adding a Layer

**Goal:** add a new layer to the framework, wired into the build and covered by a test.

**Before you start:** finish [Tutorial 1 — Getting Started](./Getting-Started.md), so you can
build and run tests.

We will build a `Swish` activation ($\text{swish}(x) = x \cdot \sigma(x)$) because it is small
enough to see whole, but has a real gradient — so it exercises every part of the contract.

---

## The contract: what a layer must provide

Every layer inherits `Module<Backend>`. Only two methods are mandatory:

| Method | Required? | What it does |
|---|---|---|
| `forward(input, requires_grad)` | **always** | Computes the output. Caches whatever `backward()` will need. |
| `backward(grad_output)` | **always** | Given ∂Loss/∂output, returns ∂Loss/∂input. |
| `params()` | only if trainable | Returns pointers to the layer's weights. |
| `reset_state()` | only if stateful | Clears memory between sequences (SNN/LSTM). |
| `state_dict()` / `load_state_dict()` | only if saveable | Serialisation. |

An activation has no weights and no state, so we only need the first two.

### The rule that catches everyone

**`forward(requires_grad=true)` must be called before `backward()`.** There is no autograd
graph here — `forward` *manually caches* what `backward` needs. Skip the forward pass and
`backward` reads a stale or empty cache.

Look at the existing `ReLU` (`include/layers/activations/ReLU.hpp`) — the whole pattern in
ten lines:

```cpp
auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
{
    if (requires_grad)
        relu_grad = input > 0.0f;   // cache the 0/1 mask for backward
    return input.relu();
}

auto backward(const Tensor& grad_output) -> Tensor override
{
    return grad_output.multiply(relu_grad);   // chain rule, elementwise
}
```

## Step 1 — Write the header

Layers live in `include/layers/<category>/`. Create `include/layers/activations/Swish.hpp`:

```cpp
#ifndef SWISH_HPP
#define SWISH_HPP

#include "layers/base/Module.hpp"
#include "tensor/Tensor.hpp"

/**
 * @file Swish.hpp
 * @brief Swish activation: swish(x) = x * sigmoid(x).
 *
 * Derivative: swish'(x) = s + x*s*(1-s), where s = sigmoid(x).
 * We cache the input AND s so backward() does not recompute the exponential.
 *
 * There is no Tensor::sigmoid() -- sigmoid is built from the primitives the
 * backend does provide (ones/divide/exp), exactly as Sigmoid.hpp does.
 */
template <typename Backend>
struct SwishImpl : public Module<Backend>
{
    using Tensor = nn::TensorImpl<Backend>;

    Tensor cached_input;     // x, needed by backward()
    Tensor cached_sigmoid;   // s = sigmoid(x), reused by backward()

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        const Tensor ones = Tensor::ones(input.rows(), input.cols());
        const Tensor s = ones.divide(ones + (input * -1.0F).exp());
        if (requires_grad)
        {
            cached_input = input;
            cached_sigmoid = s;
        }
        return input.multiply(s);
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        // swish'(x) = s + x*s*(1-s)
        const Tensor ones = Tensor::ones(cached_sigmoid.rows(), cached_sigmoid.cols());
        const Tensor one_minus_s = ones + (cached_sigmoid * -1.0F);
        const Tensor local =
            cached_sigmoid + cached_input.multiply(cached_sigmoid).multiply(one_minus_s);
        return grad_output.multiply(local);
    }
};

#endif // SWISH_HPP
```

> **Check the Tensor API before you write — do not assume it looks like PyTorch.** There is no
> `sigmoid()`, no `rsub()`, no `pow()`. The available elementwise operations are:
> `add`, `add_scalar`, `multiply`, `multiply_scalar`, `divide`, `divide_scalar`, `exp`, `sqrt`,
> `square`, `abs`, `clamp`, `relu`, `leaky_relu`, `equal`. List them yourself with:
>
> ```bash
> grep -oE "auto [a-z_]+\(" include/tensor/Tensor.hpp | sort -u
> ```
>
> If an operation you need is genuinely missing, you must add it to **every** backend
> (XTensor, OpenCL, SYCL, Device) — the parity contract will not let you add it to just one.
> See [Tensor](../Core/Tensor.md).

## Step 2 — Register the convenience alias

Add to `include/layers/Layers.hpp`, next to the other activations:

```cpp
#include "layers/activations/Swish.hpp"
namespace nn { using Swish = SwishImpl<Backend>; }
```

## Step 3 — Write the test

Tests go in a `tests/` directory beside the code, named `*_gtest.cpp`. The single most
valuable test for a new layer is a **numerical gradient check**: it catches a wrong derivative,
which is otherwise invisible (the network still trains, just worse).

```cpp
// src/core/layers/tests/swish_gtest.cpp
TEST(SwishTest, BackwardMatchesNumericalGradient)
{
    nn::Swish layer;
    nn::Tensor x(1, 3);
    x.at(0,0) = -1.0f; x.at(0,1) = 0.5f; x.at(0,2) = 2.0f;

    layer.forward(x, /*requires_grad=*/true);
    const nn::Tensor upstream = nn::Tensor::ones(1, 3);
    const nn::Tensor analytic = layer.backward(upstream);

    // Numerical reference: plain scalar swish, no Tensor ops involved.
    auto swish = [](float v) { return v / (1.0F + std::exp(-v)); };

    constexpr float kEps = 1e-3F;
    for (int i = 0; i < 3; ++i)
    {
        const float v = x.at(0, i);
        const float numeric = (swish(v + kEps) - swish(v - kEps)) / (2 * kEps);
        EXPECT_NEAR(analytic.at(0, i), numeric, 1e-2F);
    }
}
```

Wire it into the nearest `tests/CMakeLists.txt`, following the existing pattern:

```cmake
add_executable(swish_gtest swish_gtest.cpp)
target_link_libraries(swish_gtest PRIVATE layers GTest::gtest_main)
nn_gtest_discover_tests(swish_gtest)   # always this wrapper, never gtest_discover_tests directly
```

> Use `nn_gtest_discover_tests()`, not `gtest_discover_tests()`. The wrapper adds a resource
> lock so GPU-touching tests can never run concurrently under `ctest -jN`.

## Step 4 — Build and run

```bash
cmake --build --preset=max-performance --target swish_gtest -j$(nproc)
ctest --test-dir out/build/max-performance -R SwishTest --output-on-failure
```

Remember: `-R` takes the **suite name** (`SwishTest`), not the target name (`swish_gtest`).

## Step 5 — Document it

The wiki is kept in sync with the code by policy, and a hook reminds you after source edits:

- Add the layer to [Core/Layers.md](../Core/Layers.md).
- If it introduces a new idea (not just a new formula), add a concept page under
  `Concepts/`, plus a plain-language version under `Concepts/Plain/`.
- Cite the source paper in [References.md](../References.md) if there is one.

## Checklist

1. Header in `include/layers/<category>/MyLayer.hpp`
2. Inherit `Module<Backend>`; implement `forward` + `backward` (+ `params`, `reset_state`,
   `state_dict` only if the layer needs them)
3. Alias in `include/layers/Layers.hpp`
4. Test in the nearest `tests/`, named `*_gtest.cpp`, with a numerical gradient check
5. Register with `nn_gtest_discover_tests()`
6. Build the target, run it by **suite** name
7. Update `Core/Layers.md` (+ a `Concepts/` page if it is a new idea)

## Common mistakes

| Symptom | Cause |
|---|---|
| `backward()` returns zeros / garbage | `forward()` was called with `requires_grad=false`, so nothing was cached |
| Gradient check fails by a constant factor | Derivative formula wrong — recheck the chain rule |
| `ctest -R MyLayer` finds nothing | Used the target name instead of the GoogleTest suite name, or wrong case |
| Compiles on CPU, breaks on GPU | Used a Tensor op that only XTensor implements |
| Stateful layer leaks between sequences | `reset_state()` not implemented, or not clearing every member |

## Next

- [Core/Layers.md](../Core/Layers.md) — every existing layer
- [Tensor](../Core/Tensor.md) — the operations available to you
- [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md) — for spiking
  layers, where `backward` needs a surrogate because the spike is non-differentiable

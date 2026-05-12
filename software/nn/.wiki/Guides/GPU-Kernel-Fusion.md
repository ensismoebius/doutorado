# GPU Kernel Fusion

Two complementary strategies for reducing OpenCL kernel launch overhead. AMD Renoir APU
has ~10 µs launch overhead; for small tensors (batch=32, features=256→64) that dominates
compute by 500×. Eliminating launches is the primary lever.

---

## Strategy A — Hand-fused kernels (implemented)

Manually identify the most common `matmul → activation` sequences and write a single
OpenCL kernel that performs both in one GPU pass. No intermediate global-memory write.

### Implemented kernels (in `KERNEL_SOURCE_FUSED`)

| Kernel | Replaces | Speedup (Renoir, batch=32, 256→64) |
|---|---|---|
| `matmul_rhs_transposed_bias_relu_kernel` | `matmul_rhs_transposed_bias` + `relu_kernel` | **1.70×** (measured) |
| `matmul_rhs_transposed_bias_leaky_relu_kernel` | `matmul_rhs_transposed_bias` + `leaky_relu_kernel` | ~1.70× |
| `matmul_rhs_transposed_bias_sigmoid_kernel` | `matmul_rhs_transposed_bias` + `sigmoid_kernel` | ~1.70× |
| `matmul_rhs_transposed_bias_tanh_kernel` | `matmul_rhs_transposed_bias` + `tanh_kernel` | ~1.70× |

Benchmark method: 10 warmup + 100 timed iterations, 4×8 input, 6×8 weight, 6×1 bias.
Unfused: 1198 µs/iter → Fused: 704 µs/iter → **1.70× speedup** (2026-05-12, AMD Renoir APU).

### Backend methods

```cpp
// OpenCLTensorBackend — all column-major (row + col*M indexing)
matmul_transposed_add_col_bias_relu(W, b)
matmul_transposed_add_col_bias_leaky_relu(W, b, alpha)
matmul_transposed_add_col_bias_sigmoid(W, b)
matmul_transposed_add_col_bias_tanh(W, b)
```

### Status (2026-05-12)

Fused kernels are implemented and tested. `Trainer` now wraps each mini-batch in a
`BatchScope` (`#if defined(NN_BACKEND_OPENCL)`), so the ref-counted deferred-sync
from `OpenCLContext` is activated automatically — no per-layer changes required.
User-reported: **"made a huge difference"** in GPU utilisation on first production run.

See [GPU-Saturation.md](GPU-Saturation.md) for the `BatchScope` ref-count fix details.

### How to use from a layer

```cpp
// In LinearReLUImpl::forward() or ResidualBlock::forward()
if constexpr (requires(const Backend& in, const Backend& w, const Backend& b) {
    in.matmul_transposed_add_col_bias_relu(w, b);
}) {
    result = Tensor(input.get_backend().matmul_transposed_add_col_bias_relu(
        weight.get_backend(), bias.get_backend()));
} else {
    result = Tensor(input.get_backend().matmul_transposed_add_col_bias(
        weight.get_backend(), bias.get_backend())).relu();
}
```

### What is NOT fusible with this approach

- Two consecutive matmuls — K-loop is sequential per work-item; no way to merge
- `lif_step_kernel` — stateful (reads/writes `v_mem`); must stay separate
- `mse_kernel`, `sum_kernel` — work-group barrier reductions; split fusion boundary
- LSTM gate matmul — one combined `[B, 4H]` matmul feeds sigmoid/tanh on column
  slices, not separate matmul-per-gate outputs

---

## Strategy B — Whole-network tracing + JIT codegen (future)

### Concept

Instead of eager kernel dispatch, run the forward/backward pass in **recording mode**:
ops push `OpNode` records into a `ComputeGraph` DAG instead of launching kernels.
The graph is then partitioned, code-generated into OpenCL C, compiled, cached by graph
hash, and executed as one (or a few) mega-kernels.

This is exactly what XLA (TensorFlow/JAX) and `torch.compile` (Triton) do.

### Architecture required

#### 1. `ComputeGraph` IR

```cpp
struct OpNode {
    std::string op;           // "matmul", "relu", "add_bias", "leaky_relu", ...
    std::vector<int> inputs;  // indices into nodes list
    std::vector<float> params; // alpha, scalar values
    std::array<size_t, 2> shape_out;
    int node_id;
};

struct ComputeGraph {
    std::vector<OpNode> nodes;
    int record(std::string op, std::vector<int> inputs,
               std::vector<float> params, std::array<size_t,2> shape);
};
```

#### 2. Recording mode on `OpenCLTensorBackend`

Add `static thread_local ComputeGraph* active_graph_`. When non-null, every backend
method pushes an `OpNode` and returns a "symbolic" backend holding a `node_id` instead
of executing. When null, existing eager path runs unchanged.

```cpp
OpenCLTensorBackend OpenCLTensorBackend::relu() const {
    if (active_graph_) {
        int id = active_graph_->record("relu", {m_node_id}, {}, shape_);
        return OpenCLTensorBackend::symbolic(id, shape_);
    }
    // ... existing eager path ...
}
```

#### 3. Graph partitioner

Topological walk. Cut at:
- Any `matmul` node (K-loop boundary)
- Any reduction node (`sum`, `mse`)
- Any shape-changing node (`transpose`, `reshape`)

Each contiguous run of element-wise ops between cuts → one `FusedRegion`.

#### 4. OpenCL C code generator

```cpp
std::string codegen_fused_region(const std::vector<OpNode>& region,
                                  const ComputeGraph& graph) {
    // Emit one kernel; all ops become expressions using register variables:
    // __kernel void fused_N(__global const float* in0, ..., __global float* out) {
    //   uint idx = get_global_id(0);
    //   float v0 = in0[idx];           // input
    //   float v1 = v0 + bias[idx%N];   // add_bias
    //   float v2 = v1 > 0.f ? v1 : 0.f; // relu
    //   out[idx] = v2;
    // }
}
```

No intermediate global memory writes — intermediate values live in registers.

#### 5. Cache by graph hash

```cpp
std::string hash = sha256(graph.canonical_string());
if (!compiled_cache_.count(hash))
    compiled_cache_[hash] = km.compile_from_source(codegen(graph));
return compiled_cache_[hash];
```

First call per unique network topology pays JIT cost (~200 ms on Renoir, one-time).
All subsequent calls use the cached binary — zero recompilation.

#### 6. Execution

For each `FusedRegion`: bind input/output buffers, launch the generated kernel.
For matmul nodes: use the existing `matmul_rhs_transposed_bias_kernel` (already optimal).

### Fusion barriers (can never cross)

| Op | Reason |
|---|---|
| `matmul` | K-dimension sequential loop; different work-item output per row×col |
| `sum_kernel` / `mse_kernel` | Work-group barrier tree reduction |
| `transpose` | Permutes memory layout — requires own pass |
| `lif_step_kernel` | Reads/writes `v_mem` state buffer; stateful |
| `reshape` | Metadata-only but breaks index arithmetic |

### Realistic scope (one developer)

| Phase | Effort |
|---|---|
| `ComputeGraph` IR + recording mode | 1–2 weeks |
| Graph partitioner (topo sort + cut) | 1 week |
| Codegen for element-wise ops | 1 week |
| Compile/cache pipeline in KernelManager | 3 days |
| Correct matmul boundary handling | 1 week |
| Backward pass tracing (grad graph derived) | 2 weeks |
| Correctness testing across all layer combos | 2 weeks |
| **Total** | **~2–3 months** |

### Benefit at full implementation

For a 3-layer SNN encoder (Linear→LIF→Linear→LIF→Linear→identity):
- Today: 6 kernel launches (3 matmul + 3 lif_step) + queue sync overhead
- With tracer: 3 matmul kernels + 3 lif_step kernels (lif is not fusible), but all
  element-wise post-matmul ops (bias, any non-stateful activation) collapse to zero
  extra launches
- For ResNet with Linear→ReLU×N: reduces to N matmul kernels, all activations free

### Key references

- XLA HLO (High Level Optimizer): google.github.io/xla/
- `torch.compile` / Triton: openai.com/research/triton
- Halide scheduling language (same concept, image processing domain)

---

## Choosing the right strategy

| Scenario | Use |
|---|---|
| Known fixed activation (ReLU, sigmoid, tanh) after matmul | Strategy A — one-line kernel addition |
| Dynamic or user-configurable activation | Strategy B tracer |
| LSTM gates (sigmoid/tanh on column slices of shared pre) | Neither directly — LSTM is already one combined matmul |
| SNN LIF layers | Neither — LIF is stateful, not fusible |
| Reduce training time by ≥2× on APU-class hardware | Strategy B worth implementing |

# OpenCL Debugging and Performance

How to profile, debug and optimise the OpenCL backend on this project's hardware
(AMD Renoir/Lucienne iGPU, Mesa rusticl), and the hazards to avoid.

Everything here is measured, not inferred. Dates refer to when the measurement
was taken; re-measure before trusting a number on different hardware or a newer
Mesa.

---

## TL;DR — the safety rules

1. **Never disable `CL_QUEUE_PROFILING_ENABLE`.** It is the single biggest
   speed-up available (~95 µs/enqueue) and it *corrupts memory* on rusticl. The
   default is ON; only `NN_OPENCL_UNSAFE_FAST_QUEUE=1` turns it off.
2. **Debug with `RUSTICL_ENABLE=llvmpipe`**, never on the GPU. The iGPU is also
   the display adapter — a driver fault takes the desktop down and forces a
   reboot.
3. **Don't run OpenCL suites in a loop on the GPU.** Two forced reboots came
   from exactly that.
4. **For real experiment runs, prefer the CPU build.** On this hardware it is
   dramatically faster than OpenCL for the project's workloads (numbers below).

---

## 1. `RUSTICL_ENABLE=llvmpipe` — safe OpenCL debugging

### What it is

Mesa's rusticl can expose **llvmpipe**, a software (CPU) Gallium driver, as a
real OpenCL 3.0 device. You get a genuine OpenCL implementation — kernel
compilation, buffers, transfers, events, the whole API — running entirely on the
CPU, with the GPU untouched.

### Why it matters here

The compute device is the display adapter. Any rusticl fault on the GPU can kill
your session. llvmpipe removes that risk completely, so you can run crashing
workloads in a loop, under a debugger, or under valgrind, as many times as you
like.

It is also a **second independent implementation**: code that passes on both
llvmpipe and radeonsi is very likely correct, and a bug that reproduces on
llvmpipe is *yours*, while one that only appears on radeonsi is the GPU driver's.
That distinction is what cracked the profiling-queue bug below.

### How to use it

```bash
# Confirm the CPU device is exposed
RUSTICL_ENABLE=llvmpipe clinfo -l
#   Platform #0: rusticl
#    `-- Device #0: llvmpipe (LLVM 22.1.8, 256 bits)

# Run any OpenCL-built binary on the CPU device
RUSTICL_ENABLE=llvmpipe ./out/build/max-performance-opencl/path/to/binary

# Compare against the GPU device (this DOES touch the display adapter)
RUSTICL_ENABLE=radeonsi ./...
```

### Gotchas

- **`OCL_ICD_VENDORS` wants a directory, not a file.** Passing the `.icd` path
  (`OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd`) silently yields *zero*
  platforms and no error. Just use `RUSTICL_ENABLE` and leave the ICD loader
  alone.
- **llvmpipe is slow.** Suites that take seconds on the GPU take minutes. Use
  `--gtest_filter` to target one test; don't run whole suites unless you must.
- **Timings from llvmpipe are meaningless.** Correctness results are not.

### Other useful rusticl variables

| Variable | Effect |
|---|---|
| `RUSTICL_ENABLE` | Which Gallium drivers expose CL devices (`llvmpipe`, `radeonsi`, …) |
| `RUSTICL_DEBUG=sync` | Wait on the GPU after every event (serialises heavily) |
| `RUSTICL_DEBUG=memory` | Memory-object debugging |
| `RUSTICL_DEBUG=validate` | Validate internally generated SPIR-V |
| `RUSTICL_MAX_WORK_GROUPS` | Cap work-groups per dimension — splits long dispatches, improves desktop responsiveness during compute |
| `RUSTICL_DEVICE_TYPE` | Override reported device type |

Mesa's own docs note rusticl is not enabled by default because doing so "can
impact system stability until remaining core issues are ironed out." Treat it as
a young driver.

---

## 2. Profiling technique — LD_PRELOAD shim around the OpenCL API

Guessing where OpenCL time goes is unreliable. `perf` may not be installed, and
it attributes time to the driver rather than to the API call that caused it. A
tiny interposer gives an exact per-call breakdown.

Save as `clshim.c`, build with
`gcc -O2 -fPIC -shared clshim.c -o clshim.so -ldl`:

```c
#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <dlfcn.h>
#include <stdio.h>
#include <time.h>

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
                        return t.tv_sec*1e3+t.tv_nsec/1e6;}
#define N 6
static const char* NAMES[N]={"clEnqueueNDRangeKernel","clEnqueueReadBuffer(blocking)",
  "clEnqueueReadBuffer(async)","clEnqueueWriteBuffer(blocking)",
  "clEnqueueWriteBuffer(async)","clFinish"};
static long cnt[N]; static double ms[N];
static void rec(int i,double d){cnt[i]++;ms[i]+=d;}

__attribute__((destructor)) static void dump(void){
  double tot=0; for(int i=0;i<N;i++) tot+=ms[i];
  fprintf(stderr,"\n=== OpenCL API time ===\n");
  for(int i=0;i<N;i++) if(cnt[i])
    fprintf(stderr,"%-32s calls=%-9ld total=%9.1f ms  avg=%7.1f us\n",
            NAMES[i],cnt[i],ms[i],ms[i]*1000/cnt[i]);
  fprintf(stderr,"%-32s %28.1f ms\n","TOTAL in OpenCL API",tot);
}

cl_int clEnqueueNDRangeKernel(cl_command_queue q,cl_kernel k,cl_uint wd,
    const size_t*o,const size_t*g,const size_t*l,cl_uint n,const cl_event*e,cl_event*ev){
  static cl_int(*r)(cl_command_queue,cl_kernel,cl_uint,const size_t*,const size_t*,
                    const size_t*,cl_uint,const cl_event*,cl_event*);
  if(!r)r=dlsym(RTLD_NEXT,"clEnqueueNDRangeKernel");
  double t=now(); cl_int rc=r(q,k,wd,o,g,l,n,e,ev); rec(0,now()-t); return rc;}

cl_int clEnqueueReadBuffer(cl_command_queue q,cl_mem b,cl_bool blk,size_t off,
    size_t sz,void*p,cl_uint n,const cl_event*e,cl_event*ev){
  static cl_int(*r)(cl_command_queue,cl_mem,cl_bool,size_t,size_t,void*,cl_uint,
                    const cl_event*,cl_event*);
  if(!r)r=dlsym(RTLD_NEXT,"clEnqueueReadBuffer");
  double t=now(); cl_int rc=r(q,b,blk,off,sz,p,n,e,ev); rec(blk?1:2,now()-t); return rc;}

cl_int clEnqueueWriteBuffer(cl_command_queue q,cl_mem b,cl_bool blk,size_t off,
    size_t sz,const void*p,cl_uint n,const cl_event*e,cl_event*ev){
  static cl_int(*r)(cl_command_queue,cl_mem,cl_bool,size_t,size_t,const void*,
                    cl_uint,const cl_event*,cl_event*);
  if(!r)r=dlsym(RTLD_NEXT,"clEnqueueWriteBuffer");
  double t=now(); cl_int rc=r(q,b,blk,off,sz,p,n,e,ev); rec(blk?3:4,now()-t); return rc;}

cl_int clFinish(cl_command_queue q){
  static cl_int(*r)(cl_command_queue);
  if(!r)r=dlsym(RTLD_NEXT,"clFinish");
  double t=now(); cl_int rc=r(q); rec(5,now()-t); return rc;}
```

Usage: `LD_PRELOAD=./clshim.so ./your_binary 2> profile.txt`

**Caveat:** the shim inflates absolute wall time substantially (a 74 s run became
199 s). Trust the **call counts** and the **relative** breakdown; do not quote
its absolute milliseconds as the cost of the unshimmed run.

Example output from an Guayaquil run — the result that redirected the whole
optimisation effort:

```
clEnqueueWriteBuffer(blocking)  516993 calls   95986 ms  (50%)
clEnqueueReadBuffer(blocking)   227452 calls   66910 ms  (35%)
clFinish                        123818 calls   24535 ms  (13%)
clEnqueueNDRangeKernel          678160 calls    4534 ms  ( 2.4%)
```

Kernels were 2.4% of OpenCL time. Any plan to speed things up by tiling the GEMM
kernels was therefore pointless, and was abandoned on the strength of this table.

---

## 3. Microbenchmarking per-call costs

Before optimising, measure what a single OpenCL call actually costs on the target
device. A standalone C program (no project dependencies) is the fastest way.

Measured on Renoir iGPU / rusticl, 2026-07-18:

| Operation | Cost |
|---|---|
| `clEnqueueNDRangeKernel`, profiling queue | ~100 µs |
| `clEnqueueNDRangeKernel`, non-profiling queue | ~4 µs |
| `clFinish` | ~90 µs |
| Blocking `clEnqueueWriteBuffer`, 1 KiB | ~43 µs |
| Non-blocking `clEnqueueWriteBuffer`, 1 KiB | ~2.5 µs |
| Blocking `clEnqueueReadBuffer`, 1 KiB | ~231 µs |
| Non-blocking `clEnqueueReadBuffer`, 1 KiB | ~164 µs |
| `clCreateBuffer` + release, 1 KiB | ~1.6 µs |

Consequences:

- **Reads are expensive no matter what.** The only real fix is not reading back.
- **Enqueue overhead dominates small work.** Anything under ~10⁵ FLOP per kernel
  is latency-bound, not compute-bound.

Naive GEMM throughput, same device:

| M×N×K | GFLOP/s |
|---|---|
| 1 × 64 × 256 | 0.37 |
| 1 × 256 × 256 | 0.73 |
| 32 × 64 × 256 | 14.0 |
| 128 × 256 × 256 | 31.8 |

**The GPU/CPU crossover is around M ≥ 32 rows.** At batch = 1 with 256-wide
layers the iGPU loses badly to OpenBLAS — 288 µs vs 11 µs for 1×256×64.

---

## 4. The `CL_QUEUE_PROFILING_ENABLE` hazard

### Symptom

Nondeterministic `free(): double free detected in tcache 2`, SIGSEGV and SIGABRT
inside `libRusticlOpenCL.so.1`; on the GPU, the display dies and the machine
needs a hard reboot. Tests in `thesis_classifiers_gtest` fail in a shifting,
apparently random subset.

### Cause

Creating the command queue **without** `CL_QUEUE_PROFILING_ENABLE` exposes a
latent bug in rusticl's host-side event bookkeeping. The profiling path appears
to hold an extra reference that masks a premature free.

It is **not** a GPU-completion race: `RUSTICL_DEBUG=sync` (wait on GPU after
every event) does **not** help. It is host-side.

### Evidence

`thesis_classifiers_gtest --gtest_filter=ThesisRunClassifier.DsnnWithRegularizationRuns`,
on the **llvmpipe CPU device** — so this is a driver bug reproducible with no GPU
at all:

| Configuration | Result |
|---|---|
| Unmodified baseline (profiling on) | 6/6 pass |
| Profiling **off** | **0/6 pass** |
| Profiling **on** | 6/6 pass |
| Profiling off + `RUSTICL_DEBUG=sync` | 0/6 pass |

Under valgrind the test **passes with 0 errors** — valgrind's allocator and
thread serialisation hide it. Absence of valgrind findings here is not evidence
of correct code.

### Current handling

`OpenCLContext` creates the queue with profiling **on by default**.
`initialize_runtime_or_throw` can only ever turn it *on*, never off, so no caller
can silently re-arm the hazard. `NN_OPENCL_UNSAFE_FAST_QUEUE=1` opts into the
fast path and logs a warning.

---

## 5. Debugging methodology that worked

A reproducible recipe for "OpenCL is crashing and I don't know why":

1. **Get a fast repro.** Narrow to one test with `--gtest_filter`. This one
   crashed in ~1 s, which made everything afterwards cheap.
2. **Move to llvmpipe.** Removes the display risk and tells you immediately
   whether the bug is yours or the GPU driver's.
3. **Read the coredump, don't guess.**
   ```bash
   coredumpctl list
   coredumpctl debug --debugger=gdb -1 --debugger-arguments="-batch -ex 'bt 30'"
   ```
   Frames entirely inside `libRusticlOpenCL.so.1` ⇒ driver. Frames in project
   code ⇒ yours.
4. **Check for a stale-ABI artefact before believing a backtrace.** Adding a
   member to a class that crosses a shared-library boundary (e.g.
   `OpenCLTensorBackend` in `libstatistics.so`) gives bizarre crashes if only
   some targets were rebuilt. Compare artefact mtimes against your header edit,
   and do a full `cmake --build <dir>` — not `--target X` — before trusting any
   crash.
5. **Run valgrind, but interpret it carefully.** `valgrind --error-limit=no
   --num-callers=25`. A *clean* result does not exonerate the code; it suggests a
   race or a driver-side write.
6. **Bisect by reverting one change at a time**, rebuilding fully between steps,
   with N≥3 repeats per configuration. Establish the baseline flake rate first —
   otherwise you cannot tell a regression from existing noise.
7. **Verify the fix with a decisive A/B**, N≥6 both ways.

### Measuring safely and honestly

- **Delete the results directory between runs.** Guayaquil/Thesis resume from
  `results/checkpoints/`, keyed by config hash; a stale checkpoint produces a
  13-second "run" with numbers identical to the previous one. This silently
  invalidated one whole comparison before it was caught.
- **Repeat every timing ≥ 2×.** Run-to-run spread is real.
- **Rebuild fully between A and B.** `--target` builds leave mixed artefacts.
- Use `git stash` + a `trap ... EXIT` to guarantee the working tree is restored
  even if the machine dies mid-run.

---

## 6. Measured results (Guayaquil, 6 train / 4 val / 2 epochs, window 256)

| Configuration | Wall | LSTM train |
|---|---|---|
| OpenCL baseline | 190 s, 194 s | ~142.6 s |
| OpenCL, all optimisations, **unsafe** queue | 73–75 s | ~50.4 s |
| OpenCL, all optimisations, **safe** queue | 195 s | 144.8 s |
| OpenCL, safe queue + LSTM framing | **35 s** | **18.3 s** |
| CPU (xtensor/OpenBLAS), framing off | 5 s | 3.7 s |
| CPU (xtensor/OpenBLAS), framing on | **1 s** | **0.48 s** |

Read carefully:

- The 2.6× OpenCL speed-up came **almost entirely from disabling queue
  profiling**, which is unsafe. With the safe default, the backend-level
  optimisations (fewer readbacks, async uploads, no per-forward weight copies,
  device-side view ops) are **performance-neutral** on this driver.
- The LSTM framing change is worth ~8× on **both** backends and is the only
  large, safe win found.
- **The CPU backend remains ~35× faster than OpenCL** for this workload. On this
  hardware, run experiments on `max-performance`, not `max-performance-opencl`.

### Why device-side view ops are off by default

`slice_time` / `set_time_slice` / `setBlock` / `block` / `row` / `col` /
`topRows` / `leftCols` can run as a device kernel (`strided_copy_2d_kernel`) or
as a host element loop. Which wins depends entirely on the per-enqueue cost:

- Profiling **off** (~4 µs/enqueue): the device path wins.
- Profiling **on** (~95 µs/enqueue, the safe default): the device path *loses* —
  it replaces a cheap host memcpy of a small slice with a kernel launch, raising
  enqueue count from 678k to 817k (≈ +13 s on the benchmark above).

So the default matches the default queue mode: **off**. Enable with
`NN_OPENCL_DEVICE_VIEW_OPS=1` on any stack where enqueues are cheap. Both paths
are covered by `OpenCLViewOpsTest` in `opencl_tensor_backend_gtest`.

---

## 7. LSTM input framing (`model.lstm_frame_size`)

The largest safe win, and it is a modelling fix rather than a backend one.

Guayaquil previously built the LSTM autoencoder with `input_size = 1` and
`seq_len = window_size`, i.e. a 256-sample window was consumed **one scalar per
timestep**. The dominant cost is the recurrent term `h·Uᵀ` (U is `(4H, H)`,
16 384 MACs for H=64), and it was paid 256 times.

`model.lstm_frame_size` (default **8**) groups that many consecutive samples into
each timestep:

| | frame=1 (old) | frame=8 |
|---|---|---|
| sequential steps | 256 | 32 |
| total MACs | 8 523 840 | 1 184 256 |
| CPU LSTM train | 3 711 ms | 478 ms |
| GPU LSTM train | 144 848 ms | 18 335 ms |

~7× fewer MACs and ~8× less sequential depth, discarding no information.
`lstm_frame_size` must divide `dataset.window_size` (validated in
`GuayaquilConfig::validate`).

**Implementation note — framing is not a plain reshape.** Storage is
column-major, so reshaping `(256,1)` to `(32,8)` would put samples
`{t, t+32, t+64, …}` in frame `t` (a polyphase split), not consecutive samples.
`to_lstm_frames()` reshapes to `(frame, T)` and transposes, which yields
`element(t,d) = sample[t*frame + d]` as intended.

**Scientific note.** This changes the LSTM architecture and therefore the paper's
LSTM-AE results. It is arguably a *fairer* baseline: the SNN-AE sees all 256
samples at once through `linear:64`, while the old LSTM saw one scalar at a time
across 256 steps — a regime where LSTMs are known to struggle with long-range
dependencies. Set `lstm_frame_size: 1` to reproduce the original behaviour.

---

## See also

- [GPU Saturation](./GPU-Saturation.md)
- [GPU Kernel Fusion](./GPU-Kernel-Fusion.md)
- [LSTM Performance](./LSTM-Performance.md)
- [Memory Diagnostics](./Memory-Diagnostics.md)
- [Tensor](../Core/Tensor.md)

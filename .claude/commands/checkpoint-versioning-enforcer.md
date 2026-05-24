
# checkpoint-versioning-enforcer

Goal
- Ensure model checkpoints are self-describing: they carry the format version and architecture shape needed to detect and reject incompatible loads.

Rules

- RULE: VERSION_HEADER
  DO: Every checkpoint must begin with a format version string (e.g., `"nn_checkpoint_v1"`)
  AVOID: No loading a checkpoint without verifying this header first
- RULE: ARCHITECTURE_METADATA
  DO: Each checkpoint must include layer names, shapes (in, out), and dtypes
  AVOID: No loading weights into a model whose architecture was not validated against the saved metadata
- RULE: FAIL_ON_MISMATCH
  DO: When loading, if architecture metadata does not match the current model, fail loudly with a diagnostic
  AVOID: No silent loading of incompatible weights
- RULE: PROJECT_VERSION_TAG
  DO: Include the project CMake version (`0.2.0` or current) in the checkpoint metadata. Warn when checkpoint version predates current code
- RULE: ATOMIC_WRITE
  DO: Write checkpoints atomically (write to `<path>.tmp`, then rename)
  AVOID: No partial checkpoint files that can corrupt a training run
- RULE: COMPANION_JSON
  DO: Alongside the binary weight file, always write a `<name>.meta.json` with version, architecture, seed, and timestamp
  AVOID: No binary-only checkpoints

Validation

- Loading a checkpoint from a different architecture version prints an error and does not silently proceed.
- `model.meta.json` is present alongside every `model.bin`.
- Atomic write: no partial `.bin` file left on crash.

Project Context (nn framework)

**Checkpoint naming pattern:** `results/checkpoints/article_{model}_{backend}_{dataset}_{run_tag}.json`
Example: `results/checkpoints/article_lstm_ae_xtensor_fsdd_r01.json`

**Two serialization APIs:**
- `NetworkSerializer` (`include/nn/saver/NetworkSerializer.hpp`) — full `state_dict` map → `.npz` file; preferred for new code
- `NnSaver` (`include/nn/saver/NnSaver.hpp`) — legacy weight+bias pair → `_weights.npy` + `_bias.npy`; do not use for new layers

**Load pattern:**
```cpp
model.load_state_dict(NetworkSerializer::load("model.npz"));
```
Validate architecture metadata from the checkpoint against the current model before loading.

**Wiki & knowledge graph:**
- Documentation at `.wiki/` — theory, guides, experiment pages, concept definitions
- Graph output at `.wiki/graphify-out/` — 1926 nodes, 4987 edges, 203 communities
- Find any symbol/concept:
```bash
python3 -c "
import json,sys
with open('.wiki/graphify-out/graph.json') as f: g=json.load(f)
q=sys.argv[1].lower()
for n in g['nodes']:
    if q in n['id'].lower() or q in n.get('label','').lower():
        print(n['id'],'|',n.get('source_file',''),'|',n.get('source_location',''))
" <QUERY>
```
- Workflow: `GRAPH_REPORT.md` → community → node → `source_file` → read → follow edges

Expected Checkpoint Layout

```
checkpoints/<experiment_id>/<timestamp>/
├── model.bin           ← weight tensors
└── model.meta.json     ← { "format_version": "nn_checkpoint_v1",
                             "project_version": "0.2.0",
                             "timestamp": "...",
                             "random_seed": 42,
                             "layers": [{"name": "fc1", "shape": [128, 64], "dtype": "float32"}] }
```

Key Files to Update

- [include/nn/saver/NnSaver.hpp](include/nn/saver/NnSaver.hpp) — add version + metadata write
- [include/nn/saver/NetworkSerializer.hpp](include/nn/saver/NetworkSerializer.hpp) — add load-time compatibility check

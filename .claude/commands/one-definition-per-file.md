
# one-definition-per-file

Goal
- Keep the codebase modular: each type in its own file. Improves build parallelism and reduces header dependency chains.

Rules

- RULE: ONE_CLASS_PER_FILE
  DO: Each class must reside in its own header file. Exception: tightly coupled RAII handle + manager pairs may stay together
- RULE: ONE_ENUM_PER_FILE
  DO: Each `enum class` must reside in its own header file
- RULE: ONE_INTERFACE_PER_FILE
  DO: Each interface (abstract class with pure virtual methods) must have its own header
- RULE: AGGREGATE_HEADERS
  DO: Create aggregation headers that `#include` all related type headers. This maintains backward compatibility for existing includes while enabling granular includes

Validation

- Build passes after refactoring.
- All tests pass.
- No existing `#include` path breaks (aggregation header in place).

Project Context (nn framework)

**Reference layout** (one class per file, followed correctly):
- `include/nn/layers/spiking/` — one LIF variant per file: `Leaky.hpp`, `LeakyBPTT.hpp`, `ThresholdDependentBatchNorm.hpp`, `PoissonLatentLayer.hpp`
- `include/nn/layers/activations/` — one activation per file: `ReLU.hpp`, `Sigmoid.hpp`, `Tanh.hpp`
- `include/nn/layers/spiking/ExponentialSurrogate.hpp`, `BoxcarSurrogate.hpp` — one surrogate per file

**Anti-pattern (already fixed):** Multiple surrogate gradient types in a single header. This was previously the case — do not revert.

**Exception:** `include/nn/layers/Layers.hpp` is auto-generated and aggregates all layers; it is `#include`-only, not for editing. It is `.gitignore`d.

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

Naming Conventions

| Kind | Pattern | Example |
|------|---------|---------|
| Interface | `I<Name>.hpp` | `ISurrogateGradient.hpp` |
| Aggregation | Plural/collective name | `Transforms.hpp` |
| Concrete class | Original name | `BoxcarSurrogate.hpp` |

Refactoring Pattern

1. Create a new separate header for each type.
2. Move the type definition to its own file.
3. Update the aggregation header to `#include` all new files.
4. Ensure all existing includes still resolve (backward compatibility).
5. Build and run all tests.

Past Refactoring Examples (reference)

- `SurrogateGradient.hpp` → `ISurrogateGradient.hpp`, `ExponentialSurrogate.hpp`, `BoxcarSurrogate.hpp`
- `Regularization.hpp` → `IRegularization.hpp`, `L1Regularization.hpp`, `L2Regularization.hpp`
- `Transforms.hpp` → `ITransform.hpp`, `Compose.hpp`, `AudioMeanStdNormalize.hpp`, `EEGWindowZScore.hpp`, `FusedModalityTransform.hpp`
- `Device.hpp` → `DeviceType.hpp`, `Device.hpp`, `DeviceRuntime.hpp`
- `Trainer.hpp` → `EpochResult.hpp`, `TrainerConfig.hpp`, `Trainer.hpp`

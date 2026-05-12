---
name: one-definition-per-file
description: "Enforce one enum/class/interface per header file for better modularity and compile times."
---

# one-definition-per-file

Goal
- Keep codebase modular with each type in its own file.
- Improve build parallelism and reduce header dependency chains.

Rules
- RULE: ONE_CLASS_PER_FILE
  DO: Each class must reside in its own header file.
  AVOID: Multiple class definitions in a single header.
  EXCEPTION: Related helper/impl structs that are tightly coupled (e.g., RAII handle + manager) may stay together.
- RULE: ONE_ENUM_PER_FILE
  DO: Each enum class must reside in its own header file.
  AVOID: Multiple enum definitions in a single header.
- RULE: ONE_INTERFACE_PER_FILE
  DO: Each interface (abstract class with pure virtual methods) must have its own header.
  AVOID: Mixing interface definitions with implementations.
- RULE: AGGREGATE_HEADERS
  DO: Create aggregation headers that include all related type headers (e.g., Transforms.hpp includes ITransform.hpp, Compose.hpp, etc.).
  PURPOSE: Maintain backward compatibility for existing includes while enabling granular includes.

Naming Conventions
- Interface: Prefix with I (e.g., ISurrogateGradient.hpp, ITransform.hpp)
- Aggregation: Use plural or collective name (e.g., Transforms.hpp, SurrogateGradient.hpp)
- Keep original name for main class/impl (e.g., BoxcarSurrogate.hpp, Compose.hpp)

Refactoring Pattern
1. Create new separate header for each type
2. Move type definition to its own file
3. Update aggregation header to include all new files
4. Ensure all existing includes still work (backward compatibility)

Examples of Refactored Files
- SurrogateGradient.hpp: Split into ISurrogateGradient.hpp, ExponentialSurrogate.hpp, BoxcarSurrogate.hpp
- Regularization.hpp: Split into IRegularization.hpp, L1Regularization.hpp, L2Regularization.hpp
- Transforms.hpp: Split into ITransform.hpp, Compose.hpp, AudioMeanStdNormalize.hpp, EEGWindowZScore.hpp, FusedModalityTransform.hpp
- Device.hpp: Split into DeviceType.hpp, Device.hpp, DeviceRuntime.hpp
- DataLoader.hpp: Split into DefaultSamplerType.hpp, DataLoader.hpp
- Trainer.hpp: Split into EpochResult.hpp, TrainerConfig.hpp, Trainer.hpp

Validation
- Run build to verify includes work correctly.
- Ensure all tests pass after refactoring.

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

---
name: logging
description: "Skill for consistent use of nn::logging::Logger and migration from ad-hoc prints."
---

# logging

Goal
- Centralize runtime output and remove ad-hoc console/file diagnostics.

Rules
- RULE: LOGGER_ONLY
  DO: Use `NN_LOG_ERROR/WARN/INFO/DEBUG`.
  AVOID: New `std::cout`, `std::cerr`, `/tmp/*.log` diagnostics in core paths.
- RULE: LEVEL_DISCIPLINE
  DO: Map severity correctly (`ERROR`/`WARN`/`INFO`/`DEBUG`).
  AVOID: High-volume info logs in hot loops.
- RULE: CAPTURE_CONSISTENCY
  DO: Use `StreamRedirector` where unified output is required.
  AVOID: Mixed direct console and logger writes.
- RULE: PERF_GATE
  DO: Keep logging out of tight inner loops unless gated by debug level.
  AVOID: High-frequency logging in hot paths.

Workflow
1. Replace ad-hoc prints in target files.
2. Add `Logger.hpp` include where needed.
3. Build and run smoke path to validate output.

Validation
- No new ad-hoc prints in modified core files.
- Output remains readable with progress UI.

Project Context (nn framework)
**Logger header:** `include/nn/logging/Logger.hpp` — macros: `NN_LOG_ERROR`, `NN_LOG_WARN`, `NN_LOG_INFO`, `NN_LOG_DEBUG`

**Key log points in `Trainer.hpp`:**
- `INFO`: epoch start/end with loss values
- `DEBUG` (gated, not in hot path): per-batch loss when debug level enabled
- `ERROR`: NaN loss detected — log and abort training
- `WARN`: SNN biophysical param (R, C) hit clamp boundary

**Never log inside:**
- `LeakyBPTT` inner time loop — called `time_steps × batch_size` times per forward
- `matmul` inner K-loop — called `rows × cols × K` times
- Any loop with >1000 iterations in typical workload

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


# class-balance-and-stratification-enforcer

Goal
- Ensure that class imbalance is handled explicitly: stratified splits are actually used when declared, and evaluation metrics account for imbalance.

Rules

- RULE: STRATIFIED_MEANS_STRATIFIED
  DO: If config declares `stratified: true`, a stratified sampler must be used — not a random sampler
  AVOID: No config flag that does nothing
- RULE: CLASS_WEIGHTS_AUTO
  DO: When class counts are imbalanced (any class < 80% of majority class count), auto-compute inverse-frequency weights for `WeightedRandomSampler`
  AVOID: No manual weight tuning without documenting the imbalance ratio
- RULE: WEIGHTED_METRICS_REQUIRED
  DO: For multi-class problems, always report macro-average F1, precision, and recall alongside micro/accuracy
  AVOID: No evaluation that reports accuracy only on imbalanced datasets
- RULE: PER_FOLD_CLASS_CHECK
  DO: Before training each fold, log class counts in the train and validation splits. Warn if any class has fewer than `min_samples_per_class` (default: 5) examples in the validation fold
- RULE: STRATIFICATION_VERIFIED
  DO: After creating folds, assert that class distribution in each fold is within ±5% of the overall distribution. Log a warning if stratification failed to maintain balance

Validation

- Config `stratified: true` → stratified sampler is used (grep for sampler instantiation).
- Eval output includes macro F1, not just accuracy.
- Class counts are logged for every fold before training begins.

Project Context (nn framework)

**Dataset context:** EEG/audio windows from the 10.1117 imagined speech dataset. Windows within a session are not inherently imbalanced — class counts depend on trial design (typically balanced). Default: use `KFold`, not `StratifiedKFold`, unless label distribution check shows imbalance.

**KFold API location:** `include/nn/statistics/kfold.hpp`
- `KFold` — standard k-fold, no stratification
- `StratifiedKFold` — stratified by label; use when any class < 80% of majority
- `NestedKFold` — outer CV for model selection + inner CV for hyperparameter tuning

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

**Wiki & knowledge graph** (concepts, papers, docs — not code symbols; use the MCP tools above for those):
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

Key Files to Fix

- [src/experiments/waveletAE/WaveletAEConfig.hpp](src/experiments/waveletAE/WaveletAEConfig.hpp) — `stratified = true` field that needs enforcement
- [include/nn/dataLoaders/samplers/](include/nn/dataLoaders/samplers/) — `WeightedRandomSampler` needs class-weight auto-computation
- [include/nn/statistics/kfold.hpp](include/nn/statistics/kfold.hpp) — stratified split implementation needed
- [include/nn/statistics/multi_class_metrics.hpp](include/nn/statistics/multi_class_metrics.hpp) — add weighted/macro metric variants
- [include/nn/statistics/confusion_matrix.hpp](include/nn/statistics/confusion_matrix.hpp) — derive weighted F1 from confusion matrix

Stratified K-Fold Pattern

```cpp
// Compute class distribution before splitting:
auto class_counts = count_labels(dataset.labels());
NN_LOG_INFO("Class distribution: {}", format_counts(class_counts));

// Use stratified split (group by class, then interleave):
auto folds = stratified_kfold(dataset, config.k_folds, config.random_seed);

// Verify balance per fold:
for (auto& fold : folds) {
    assert_class_balance(fold, class_counts, tolerance=0.05f);
}
```

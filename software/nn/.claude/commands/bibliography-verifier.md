---
description: "Verify all bibliography entries in a .bib file for correctness: real publication, right journal/venue, year, volume, pages, and DOI."
---

# bibliography-verifier

Audit every `@article`, `@inproceedings`, `@book`, and `@misc` entry in one or more `.bib` files.
For each entry check: authors exist, title matches actual publication, journal/venue is correct,
year/volume/number/pages are accurate, DOI resolves to the stated paper.

## Code intelligence (MCP `code_intelligence`)

Prefer over grep/manual git/cmake for anything about the code itself:
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` / `replace_symbol` — structural findings, complexity hotspots, and hash-gated multi-site renames/edits
- `run_build` / `run_tests` / `run_lint` / `run_format` — structured build/test/lint output, not raw logs
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

## Project Context (nn framework)

**`.bib` file locations:**
- `documentation/07-articlesProduced/conference71070Guaiaquil/paper.bib` — conference paper bibliography
- `documentation/00-dissertation/monography/monography.bib` — thesis bibliography

**Domain:** neuromorphic computing, SNN, EEG/BCI, wavelet signal processing, paraconsistent logic

**Key topics to verify citations for:**
- Surrogate gradients: Neftci et al. 2019 (IEEE Signal Processing Magazine)
- LSNN: Bellec et al. 2018 (NeurIPS)
- Spike encoding (rate/latency): various IEEE TNNLS and Frontiers in Neuroscience
- Paraconsistent logic: Da Costa (foundational, mathematical logic journals)
- 10.1117 dataset: the actual SPIE proceedings entry

## Rules

- **DOI_FIRST**: Resolve DOI via `https://doi.org/<doi>` before any other source. DOI is ground truth.
- **CROSS_CHECK**: For entries without DOI, cross-check on Semantic Scholar, IEEE Xplore, ACM DL, PubMed, arXiv, and Google Scholar.
- **EXACT_MATCH**: Journal name, volume, issue, and page range must match the resolver result exactly. Abbreviated vs. full journal name is acceptable; wrong journal is not.
- **YEAR_KEY_CONSISTENCY**: If the cite key encodes a year (e.g., `smith2019fast`), verify it matches the `year` field.
- **TITLE_SUBTITLE**: Shortened titles are acceptable; fabricated titles are not. Flag if the title does not match any real publication.
- **COMPLETENESS**: Flag missing mandatory fields: `author`, `title`, `year` for all types; `journal`/`booktitle` for articles/inproceedings; `publisher` for books; `pages` for conference papers.
- **GHOST_CHECK**: Flag entries where DOI does not resolve or where no verifiable source confirms the publication exists.
- **NO_HALLUCINATION**: Do not invent corrected metadata. If unsure, mark `UNVERIFIED` and list the best candidate found with its source URL.

## Workflow

1. Locate `.bib` files: check argument, then `paper.bib`, then `*.bib` in working directory.
2. Parse all entries, extract key, type, and fields.
3. For each entry, resolve DOI if present; otherwise search by title + author + year.
4. Compare retrieved metadata against the bib entry field-by-field.
5. Classify each entry: `CORRECT` / `WRONG` / `INCOMPLETE` / `UNVERIFIED`.
6. For each non-CORRECT entry, produce the corrected bib block.
7. Apply corrections to the file only after listing all findings (ask user first if > 3 entries need changes).

## Output Format

```
## Verification Results

| # | Key | Status | Issue |
|---|-----|--------|-------|
| 1 | smith2019fast | CORRECT | — |
| 2 | jones2020deep | WRONG   | Journal: listed "Nature" actual "Proc. IEEE" |
| 3 | doe2018lstm   | INCOMPLETE | Missing pages |
```

Then for each WRONG or INCOMPLETE entry, the corrected `@article{...}` block.
Then a one-line summary: `N/M entries correct. K corrections applied.`

## Validation

- Every entry gets a status — no silent skips.
- Every WRONG entry includes the source URL confirming the correction.
- Corrected bib blocks are valid BibTeX (no encoding errors in author names, proper `--` for page ranges).
- If DOI resolver returns HTTP error or redirect loop, mark `UNVERIFIED` rather than guessing.

---
name: bibliography-verifier
description: "Verify all bibliography entries in a .bib file for correctness: real publication, right journal/venue, year, volume, pages, and DOI."
---

# bibliography-verifier

Goal
- Audit every `@article`, `@inproceedings`, `@book`, and `@misc` entry in one or more `.bib` files.
- For each entry check: authors exist, title matches actual publication, journal/venue is correct,
- year/volume/number/pages are accurate, DOI resolves to the stated paper.

Rules

- RULE: DOI_FIRST
  DO: Resolve DOI via `https://doi.org/<doi>` before any other source. DOI is ground truth
- RULE: CROSS_CHECK
  DO: For entries without DOI, cross-check on Semantic Scholar, IEEE Xplore, ACM DL, PubMed, arXiv, and Google Scholar
- RULE: EXACT_MATCH
  DO: Journal name, volume, issue, and page range must match the resolver result exactly. Abbreviated vs. full journal name is acceptable; wrong journal is not
- RULE: YEAR_KEY_CONSISTENCY
  DO: If the cite key encodes a year (e.g., `smith2019fast`), verify it matches the `year` field
- RULE: TITLE_SUBTITLE
  DO: Shortened titles are acceptable; fabricated titles are not. Flag if the title does not match any real publication
- RULE: COMPLETENESS
  DO: Flag missing mandatory fields: `author`, `title`, `year` for all types; `journal`/`booktitle` for articles/inproceedings; `publisher` for books; `pages` for conference papers
- RULE: GHOST_CHECK
  DO: Flag entries where DOI does not resolve or where no verifiable source confirms the publication exists
- RULE: NO_HALLUCINATION
  DO: Do not invent corrected metadata. If unsure, mark `UNVERIFIED` and list the best candidate found with its source URL

Workflow

1. Locate `.bib` files: check argument, then `paper.bib`, then `*.bib` in working directory.
2. Parse all entries, extract key, type, and fields.
3. For each entry, resolve DOI if present; otherwise search by title + author + year.
4. Compare retrieved metadata against the bib entry field-by-field.
5. Classify each entry: `CORRECT` / `WRONG` / `INCOMPLETE` / `UNVERIFIED`.
6. For each non-CORRECT entry, produce the corrected bib block.
7. Apply corrections to the file only after listing all findings (ask user first if > 3 entries need changes).

Validation

- Every entry gets a status — no silent skips.
- Every WRONG entry includes the source URL confirming the correction.
- Corrected bib blocks are valid BibTeX (no encoding errors in author names, proper `--` for page ranges).
- If DOI resolver returns HTTP error or redirect loop, mark `UNVERIFIED` rather than guessing.

Project Context (nn framework)

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

Output Format

```

Verification Results

| # | Key | Status | Issue |
|---|-----|--------|-------|
| 1 | smith2019fast | CORRECT | — |
| 2 | jones2020deep | WRONG   | Journal: listed "Nature" actual "Proc. IEEE" |
| 3 | doe2018lstm   | INCOMPLETE | Missing pages |
```

Then for each WRONG or INCOMPLETE entry, the corrected `@article{...}` block.
Then a one-line summary: `N/M entries correct. K corrections applied.`

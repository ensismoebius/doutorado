# Thesis Document: editing the monograph and keeping it true to the results

## The problem this page solves

The thesis PDF quotes several hundred numbers from experiments that have been re-run several
times. After one re-run, the tables (typeset from CSVs) were right, while the sentences around
them still quoted July: an EER of 0.4534 sat in the abstract next to tables that said 0.22. The
PDF compiled and nothing warned anyone. **A hand-typed number fails silently.** So the
monograph is built like code: every number and every qualitative claim about results comes
from the committed results through a script, and a stale one stops the build.

```
results/thesis/{phase00,phase01,biomex}/   (committed run outputs)
        │
        ├─ thesis_build_phase00_paraconsistent_tables.py ─► tables/phase00_*.csv ─┐
        ├─ thesis_build_phase01_auth_tables.py ───────────► tables/phase01_*.csv ─┤ csvreader
        ├─ thesis_build_biomex_tables.py ─────────────────► tables/biomex_*.csv ──┤ in ch. 9
        ├─ 03_/04_..._significance.sh ─► results/.../significance_*.csv ─┐        │
        │                                                                ▼        ▼
        └────────────────────► thesis_build_text_numbers.py ─► tables/thesis_numbers.tex ─► \tn{key}
             (+ base_facts.csv, ae_input_reference.csv)     │      in the prose
                                                            └─ CLAIMS: stops if a sentence became false
```

All paths below are relative to the repository root unless stated.

## Where things are

`documentation/00-thesis/monography/`:

| Path | What |
|---|---|
| `monografia.tex` | the root file: front matter, chapters 06–10, bibliography, appendices |
| `chapters/preamble.tex`, `chapters/data/` | class and packages (`packages.tex`), title page (`faceAndCover.tex`), **date** (`date.tex`), **keywords** (`keyWords.tex`), and `visualStyle.tex`: colours, table style, `\tn`, `\pendenteMarca` |
| `chapters/00-*.tex … 05-*.tex` | catalog record, approval sheet, dedication, thanks, epigraph, abstracts, lists, acronyms (`00-acronym.tex`) |
| `chapters/06-Introduction.tex` | context, gap and hypothesis, objectives, research questions, contributions |
| `chapters/07-bibliographicRevision.tex` | concepts used, and the state of the art (systematic review + its 2026 update) |
| `chapters/08-proposedApproach.tex` | the two bases, the strategy (Phase 00 / 01 / BIOMEX), the procedures, the simulated dysphonia |
| `chapters/09-testsAndResults.tex` | Phase 00, Phase 01, the BIOMEX-DB replication, the research questions answered |
| `chapters/10-conclusions.tex` | synthesis, contributions, limitations, future work |
| `chapters/11-appendices.tex` | defects found and fixed, reproducibility, Bayesian inference, how to build a neural network, parallelism |
| `tables/` | generated CSVs and `thesis_numbers.tex` (below); three hand-made `.tex` tables (`regularWaveletExample`, `packetWaveletExample`, `someWaveletsProperties`) |
| `images/` | figures; TikZ sources (`*.tex`), and the three figure scripts (below) |
| `listings/` | code listings quoted in the text |
| `bibliography.bib` | the bibliography (BibTeX, `abntex2` style) |
| `issues.md` | the open review items and the pending list, in Portuguese — **read it before changing the text** |
| `unesp.pdf` | the university logo used by the title page |

## Building the PDF

```bash
cd documentation/00-thesis/monography
latexmk -pdf -interaction=nonstopmode -halt-on-error monografia.tex
```

`latexmk` runs pdflatex and BibTeX as often as needed. Add `-outdir=/some/dir` to keep the
build products out of the folder (they are gitignored either way). The result must have no
undefined references (`??` in the PDF) and no overfull box above a few points. Check the log
for both. A missing `\tn` key is a compile **error**, not a warning: see below.

`monografia.pdf` is tracked. Regenerate it after a change; the author decides when a build is
committed.

## The rules of the text

| Rule | How it is enforced | What breaks if ignored |
|---|---|---|
| **No hand-typed result.** Every number that comes from a run is `\tn{key}` (decimal comma) or `\tnen{key}` (decimal point, for the English abstract) | an unknown key is a LaTeX error (`visualStyle.tex`) | **silent**: a typed number compiles and goes stale at the next re-run |
| **Every qualitative claim about results is a CLAIM.** "The same extractor wins on the three signals" has no number, yet a re-run can make it false | `CLAIMS` in `thesis_build_text_numbers.py`: each sentence as a condition on the keys; a broken one stops the script with the sentence's text | **silent** without it: every `\tn` still compiles under a false sentence |
| **Pending items are marked.** Anything not yet decided is `\pendenteMarca{[PENDENTE: …]}` (inline) or the block form in `visualStyle.tex` | highlighted in the PDF; `grep -n pendente chapters/*.tex` lists them | an unmarked placeholder reaches the defence version |
| **One visual style.** Tables: `\tableZebraStripe` before the environment, `\tableHeaderRow` on the header row, `\bottomrule`; plots in the Okabe–Ito palette of `visualStyle.tex` (cbBlue 0,114,178; cbVermillion 213,94,0; cbSkyBlue 86,180,233; cbGrey) | review | a grey or default-matplotlib figure stands out from the rest |
| **Every data-treatment choice is stated in ch. 8** (filters, exclusions, resampling, pairing rules) | review; the facts come from `base_facts.csv` | the replication cannot be judged |
| **Citations are verified** (DOI, venue, year against the publisher) before they enter `bibliography.bib` | review | a merged or invented reference (it happened in the wiki) |

### Adding a number to the text

1. Find the key: `python3 software/nn/scripts/pipeline/thesis/thesis_build_text_numbers.py <the arguments of § Results to text> --list`.
   The key families are listed in the script's docstring (`p00.*`, `p01.*`, `bx.*`, `sig.*`,
   `base.*`, `aeinput.*`).
2. If it does not exist, add it to the function of that family in the script (it reads the
   same collectors as the table scripts, so it refuses an incomplete result set), regenerate,
   and run `python3 software/nn/scripts/testing/test_thesis_phase_scripts.py`.
3. Write `\tn{the.key}` in the sentence. If the sentence also *asserts* something (an order, a
   "beats", a "no significant difference"), add the assertion to `CLAIMS` next to the others.

## Results to text

After any run that changes results (from `software/nn/`, in this order: each step reads what
the one before wrote):

```bash
# 1. tables (Phase 00 tables are also written automatically after a clean phase00 run)
python3 scripts/pipeline/thesis/thesis_build_phase00_paraconsistent_tables.py \
  --profiles-dir src/experiments/thesis/profiles/phase00 --results-dir results/thesis/phase00 \
  --tables-dir ../../documentation/00-thesis/monography/tables
python3 scripts/pipeline/thesis/thesis_build_phase01_auth_tables.py \
  --profiles-dir src/experiments/thesis/profiles/phase01 --results-dir results/thesis/phase01 \
  --tables-dir ../../documentation/00-thesis/monography/tables
python3 scripts/pipeline/thesis/thesis_build_biomex_tables.py \
  --profiles-dir src/experiments/thesis/profiles/biomex --results-dir results/thesis/biomex \
  --coretto-profiles-dir src/experiments/thesis/profiles/phase01 \
  --coretto-results-dir results/thesis/phase01 \
  --tables-dir ../../documentation/00-thesis/monography/tables

# 2. the declared significance tests
scripts/pipeline/thesis/03_thesis_phase01_significance.sh src/experiments/thesis/profiles/phase01 results/thesis/phase01
scripts/pipeline/thesis/04_thesis_biomex_significance.sh  src/experiments/thesis/profiles/biomex  results/thesis/biomex

# 3. facts about the bases and the AE-input reference (only when a base or the AE input changed)
python3 scripts/analysis/thesis_base_facts.py --biomex ~/biomex.sqlite --coretto ~/database.sqlite \
  --out results/thesis/base_facts.csv
python3 scripts/analysis/thesis_ae_input_reference.py --db ~/database.sqlite \
  --out results/thesis/ae_input_reference.csv          # needs torch; about a minute

# 4. every number and claim of the prose
python3 scripts/pipeline/thesis/thesis_build_text_numbers.py \
  --phase00-profiles src/experiments/thesis/profiles/phase00 --phase00-results results/thesis/phase00 \
  --phase01-profiles src/experiments/thesis/profiles/phase01 --phase01-results results/thesis/phase01 \
  --biomex-profiles  src/experiments/thesis/profiles/biomex  --biomex-results  results/thesis/biomex \
  --ae-input-reference results/thesis/ae_input_reference.csv --base-facts results/thesis/base_facts.csv \
  --out ../../documentation/00-thesis/monography/tables/thesis_numbers.tex

# 5. the PDF (§ Building the PDF)
```

If step 4 stops on a CLAIM, the data contradicts a sentence: **rewrite the sentence**. Do not
loosen the claim. Then commit results, tables, `thesis_numbers.tex` and the text together.

## Generated figures

| Figure | Script (in `images/`) | Regenerate with |
|---|---|---|
| `spikeEncodings.pdf` (ch. 8, the three temporal encodings) | `generate_encoding_figure.py` | `python3 generate_encoding_figure.py .` |
| `membranePotential{Full,Increase,Decay}.pdf` (ch. 7) | `generate_membrane_figures.py` | `python3 generate_membrane_figures.py .` |
| `simulatedDysphonia.pdf` (ch. 8, a real utterance degraded by the thesis code) | `generate_dysphonia_figure.py` — compiles `software/nn/src/experiments/thesis/lib/src/VoiceDegradation.cpp` with `g++` and calls it, so the figure shows what the experiment does | `python3 generate_dysphonia_figure.py --db ~/database.sqlite --out-dir .` |

Run them from `images/`. The TikZ flowcharts (`strategicStructure.tex`, `e05PipelineTwoPhase.tex`,
`e05Phase00Flow.tex`, `e05Phase01Flow.tex`, all in ch. 8) are compiled with the document.
`e05PipelineTwoPhase.tex` quotes the profile counts through `\tn{p00.n.*}`, `\tn{p01.n.total}`
and `\tn{bx.n.total}`, so it follows a change in the profile grid; the other three hold no
results, only the method, and must be redrawn by hand when the method changes.

## Still open

`issues.md` (status section at its top) holds the review items and what is left before the
defence. On 2026-11-13 the items left for the author are: the Web of Science record of the review
update (marked `\pendenteMarca` in ch. 7), the examining board, dedication / acknowledgements /
epigraph, the catalog record, and the defence date (`chapters/data/date.tex` holds a placeholder).

## Related

- [Running Thesis Profiles](./Running-Thesis-Profiles.md) — producing the results this page consumes
- [Data Bases](./Data-Bases.md) — where `base_facts.csv` comes from
- [Experiment05](../Experiments/Thesis.md) §10 — the history of the "no typed number" rule

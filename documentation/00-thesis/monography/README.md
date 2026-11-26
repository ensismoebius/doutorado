# The thesis (monograph)

*Autenticação Biométrica de Locutores Drasticamente Disfônicos Aprimorada pela Imagined
Speech* — LaTeX (`abntex2`), in Portuguese. The built PDF is `monografia.pdf` (tracked).

**No result in this folder is typed by hand.** The tables and every number of the prose are
written by scripts from the committed run outputs in `software/nn/results/thesis/`, and a
qualitative sentence that the data stops supporting stops the build. To check that, or to
re-run the experiments, follow
[Reproducing the thesis results](../../../software/nn/.wiki/Tutorials/Reproducing-the-Thesis.md).
The short version, from `software/nn/`:

```bash
scripts/pipeline/thesis/thesis_rebuild_from_results.sh --check   # tables, numbers, PDF; then compares with git
```

## What is here, and where it comes from

| Path | What | Written by |
|---|---|---|
| `monografia.tex` | the root file: front matter, chapters, bibliography, appendices | hand |
| `chapters/` | one file per chapter (`06-Introduction.tex` … `11-appendices.tex`), the front matter (`00-` … `05-`), and `preamble.tex` + `data/` (class, packages, title page, date, keywords, `visualStyle.tex`) | hand |
| `tables/*.csv` | the result tables of ch. 9 | `software/nn/scripts/pipeline/thesis/thesis_build_*_tables.py` |
| `tables/thesis_numbers.tex` | every number the prose quotes, as `\tn{key}` (decimal comma) and `\tnen{key}` (decimal point) | `software/nn/scripts/pipeline/thesis/thesis_build_text_numbers.py`, which also checks the `CLAIMS` |
| `tables/{regular,packet}WaveletExample.tex`, `someWaveletsProperties.tex` | three explanatory tables, no results | hand |
| `images/*.tex` | TikZ figures (flowcharts, diagrams), compiled with the document | hand |
| `images/spikeEncodings.pdf`, `membranePotential*.pdf`, `simulatedDysphonia.pdf` | generated figures | `images/generate_*.py` ([Thesis Document › Generated figures](../../../software/nn/.wiki/Guides/Thesis-Document.md#generated-figures)) |
| `images/` (other files) | figures from the literature or drawn by the author | — |
| `listings/` | code listings quoted in the text | hand |
| `bibliography.bib` | the bibliography; every entry verified against the publisher | hand |
| `issues.md` | open review items, the independent readings and what is left before the defence (Portuguese) | hand |

Appendix B of the thesis (`apend:reprodutibilidade`, in `chapters/11-appendices.tex`) lists the
programs and files of the experiment for a reader of the PDF; this README and the tutorial are
the same map for a reader of the repository.

## Build the PDF

```bash
cd documentation/00-thesis/monography
latexmk -pdf -interaction=nonstopmode -halt-on-error monografia.tex
```

A `\tn{key}` that `thesis_numbers.tex` does not define is a compile **error**, so a removed
result cannot leave a stale number behind. Rules for editing the text (adding a number, a
claim, a table, the visual style): [Thesis Document](../../../software/nn/.wiki/Guides/Thesis-Document.md).

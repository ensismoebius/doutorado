# documentation/

Everything written for people: the thesis, the papers, the talks, and the reading material.
The software's own documentation is the wiki, in `software/nn/.wiki/`.

| Folder | What | How to build / where to read more |
|---|---|---|
| `00-thesis/monography/` | **the thesis** (LaTeX, `abntex2`, Portuguese): `monografia.tex` + `chapters/`, generated `tables/`, `images/`, `bibliography.bib`, open items in `issues.md` | [Thesis Document](../software/nn/.wiki/Guides/Thesis-Document.md): structure, the rules (no typed numbers), regenerating tables and figures, `latexmk -pdf monografia.tex` |
| `00-thesis/presentation/` | the thesis presentation (`apresentacao.tex`, Beamer) | `latexmk -pdf apresentacao.tex` in that folder |
| `00-thesis/stateOfTheArtReview/` | the systematic review behind ch. 7 (exploratory analysis and review material) | ch. 7 of the thesis |
| `07-articlesProduced/meeting01/` | **the conference paper** *What Does Spiking Cost an Autoencoder?* (`paper.tex`, `figs/`, generated `data/`), and `review/`: the review rounds, rulings and the scripts that checked them | `pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex`; `data/` is written by `software/nn/scripts/pipeline/meeting01/` ([Meeting01](../software/nn/.wiki/Experiments/Meeting01.md)) |
| `07-articlesProduced/article/` | a tutorial article, *Understanding a Spiking Neuron* (LIF dynamics) | `article.tex` |
| `07-articlesProduced/spm-featurearticle-latex/` | the IEEE Signal Processing Magazine feature-article templates | — |
| `08-lectures/fronteiras-bitnets-redes-pulso/` | the BitNet / spiking-networks lecture (two decks from one body) | `make` in that folder; the live demos are `software/efficient_nn_lab/` (`abrir-demo.sh`) |
| `01-researchNotesAndFiles/` | research notes | — |
| `03-articlesToRead/`, `06-BooksToRead/` | the reading library (PDFs); local additions are often left untracked | — |
| `projeto_de_pesquisa.pdf` | the original research proposal | — |

Data in the papers is never typed by hand: the thesis's `tables/` and the paper's `data/` are
written by scripts from the committed results. Regenerate them, don't edit them.

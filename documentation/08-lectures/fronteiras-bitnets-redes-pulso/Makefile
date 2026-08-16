MAIN    := apresentacao
LATEXMK := latexmk
PDF     := $(MAIN).pdf

.PHONY: all pdf watch clean

all: pdf

pdf: $(PDF)

$(PDF): $(MAIN).tex preamble.tex referencias.bib slides/*.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN).tex
	rm -f $(MAIN).fls $(MAIN).fdb_latexmk

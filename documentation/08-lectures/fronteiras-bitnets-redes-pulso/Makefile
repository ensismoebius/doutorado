# Two decks share one body (conteudo.tex), gated by the \ifincludebitnet
# toggle each entry .tex sets before \input{preamble.tex} -- see
# apresentacao.tex / apresentacao-tese.tex's own comments.
FULL    := apresentacao
TESE    := apresentacao-tese
LATEXMK := latexmk

.PHONY: all pdf tese watch clean

all: pdf tese

pdf: $(FULL).pdf

tese: $(TESE).pdf

$(FULL).pdf: $(FULL).tex conteudo.tex preamble.tex referencias.bib slides/*.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(FULL).tex

$(TESE).pdf: $(TESE).tex conteudo.tex preamble.tex referencias.bib slides/*.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(TESE).tex

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(FULL).tex

clean:
	$(LATEXMK) -c $(FULL).tex
	$(LATEXMK) -c $(TESE).tex
	rm -f $(FULL).fls $(FULL).fdb_latexmk $(TESE).fls $(TESE).fdb_latexmk

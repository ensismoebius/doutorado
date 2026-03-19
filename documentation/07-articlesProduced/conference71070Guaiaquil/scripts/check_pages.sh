#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# compile twice and report page count, figures and tables
pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1 || true

if command -v pdfinfo >/dev/null 2>&1; then
  pages=$(pdfinfo paper.pdf | awk '/^Pages:/ {print $2}')
else
  # fallback: try using python PyPDF2 if available
  pages=$(python3 - <<'PY'
import sys
try:
    from PyPDF2 import PdfReader
    r=PdfReader('paper.pdf')
    print(len(r.pages))
except Exception:
    print('unknown')
PY
)
fi

figs=$(grep -c "\\begin{figure}" paper.tex || true)
tables=$(grep -c "\\begin{table}" paper.tex || true)
refs=$(grep -c "\\bibitem" paper.tex || true)

echo "pages=$pages figs=$figs tables=$tables refs=$refs"

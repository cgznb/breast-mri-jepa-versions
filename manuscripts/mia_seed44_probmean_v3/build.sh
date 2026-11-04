#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [ ! -f elsarticle.cls ]; then
  (cd vendor/elsarticle/elsarticle && latex -interaction=nonstopmode -halt-on-error elsarticle.ins)
  cp vendor/elsarticle/elsarticle/elsarticle.cls .
fi
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build_pdflatex_1.log
# BibTeX exit 1 denotes warnings, including valid conference entries without pages.
mia_bibtex_status=0
bibtex main > build_bibtex.log || mia_bibtex_status=$?
if (( mia_bibtex_status > 1 )); then
  cat build_bibtex.log
  exit "$mia_bibtex_status"
fi
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build_pdflatex_2.log
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build_pdflatex_3.log
printf '%s\n' 'Built main.pdf'

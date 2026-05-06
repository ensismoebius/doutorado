# ETCM 2026 Submission Plan

## Official Dates
- Paper submission: 10 May 2026
- Notification of acceptance: 15 June 2026
- Camera-ready submission: 26 July 2026
- Conference: 22-24 October 2026

## Current Status (as of 06 May 2026)
- Time remaining to submission: 4 days
- Build chain: pdflatex + bibtex + pdflatex + pdflatex successful
- Current length: 5 pages (within 4-6 page limit)
- Double-blind author block: enabled ("Removed for blind review")
- Experiment-backed final results: pending (depends on Experiment 04 final run)

## Execution Plan

### D-4 (Today)
- Freeze paper structure and section order.
- Remove or rewrite any sentence that overclaims results not yet produced by Experiment 04.
- Keep all result tables and plots connected to experiment export files.

### D-3
- Import final Experiment 04 outputs into article data files.
- Re-generate all figures and tables from experiment outputs only.
- Re-check metric consistency between text, tables, and plots.

### D-2
- Double-blind audit pass:
  - No author names, affiliations, emails, grant numbers, or self-identifying acknowledgements.
- Style and language pass:
  - concise wording, no exaggerated claims, no placeholders.
- Technical pass:
  - rebuild from scratch and clear fatal errors.

### D-1
- Produce submission PDF and verify final page count.
- Submit in EasyChair.
- Keep a small buffer for last-minute formatting fixes before deadline.

## Submission Checklist
- [x] English language
- [x] IEEE conference format
- [x] 4-6 pages (currently 5)
- [x] Double-blind author block in source
- [x] Reproducible LaTeX build
- [ ] Final Experiment 04 outputs integrated
- [ ] EasyChair submission completed

## Practical Rule
Do not manually type any final result value in the paper text/tables/plots. Every result must come from Experiment 04 generated files.

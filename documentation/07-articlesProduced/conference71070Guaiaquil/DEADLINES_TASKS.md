# ETCM 2026 - Deadlines And Tasks

## Official Deadlines
- Paper submission: 10 May 2026
- Notification of acceptance: 15 June 2026
- Camera-ready: 26 July 2026
- Conference dates: 22-24 October 2026

## Current Status (06 May 2026)
- Remaining time to submission: 4 days
- Current paper length: 4 pages
- Double-blind author block: enabled
- LaTeX/BibTeX chain: citations resolved; pending final data CSV integration
- Experiment 04 final result integration: in progress (waiting experiment execution)

## Task Plan By Day

### D-4 (today)
- Freeze structure and section order.
- Remove overclaims not directly supported by generated experiment outputs.
- Keep every result table/plot connected to generated data files only.

### D-3
- Run Experiment 04 final profiles.
- Export and place final result files used by the paper.
- Rebuild and verify consistency between text, tables, and plots.

### Execution Hold (current)
- Waiting for user to finish running Experiment 04 code.
- No synthetic/manual result generation will be performed.
- Next action after run completion: ingest generated CSVs, rebuild paper, and close remaining pgfplots errors.

### D-2
- Perform a full double-blind audit:
  - no names, affiliations, emails, grants, acknowledgements identifying authors.
- Perform language pass:
  - concise, direct wording; remove placeholders and vague claims.
- Rebuild from scratch and verify no fatal LaTeX/BibTeX errors.

### D-1
- Generate final submission PDF.
- Final compliance check (format, pages, blind review).
- Submit on EasyChair with time buffer before deadline.

## Submission Checklist
- [x] English language
- [x] IEEE conference format
- [x] 4-6 pages (currently 4)
- [x] Double-blind author block
- [ ] Reproducible build chain with final experiment CSVs
- [ ] Final Experiment 04 results integrated
- [ ] EasyChair submission completed

## Hard Rule
Do not manually hardcode final result numbers in the manuscript. All reported metrics must come from Experiment 04 generated files.

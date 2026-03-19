# Paper generation skeleton

Files:
- `paper.tex` — LaTeX IEEEtran skeleton
- `figs/fig1.tex` — TikZ architecture figure
- `paper.bib` — placeholder bibliography
- `scripts/check_pages.sh` — compile and check page count

To compile and check pages:
```bash
cd /home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/conference71070Guaiaquil
bash scripts/check_pages.sh
```

If the PDF has more or fewer than 6 pages, run iterative edits per the provided plan.

Measurement & simulation scripts:

 - `scripts/run_measurement.sh` — run an inference binary and record elapsed time (placeholder for energy integration).
 - `scripts/simulate_sweep.py` — generate a simulated sweep figure `figs/fig_sweep.tex` showing RMSE vs Ops for different latent dims.

Generate the sweep figure:
```bash
python3 scripts/simulate_sweep.py
```


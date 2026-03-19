#!/usr/bin/env python3
"""Generate simulated sweep data and output a TikZ figure file.
Produces: figs/fig_sweep.tex
"""
import math
from pathlib import Path

out = Path(__file__).resolve().parents[1] / 'figs' / 'fig_sweep.tex'
out.parent.mkdir(parents=True, exist_ok=True)

# Simulated sweep: latent dims and spike penalty values
latent = [32, 64, 128]
lambda_vals = [1e-5, 1e-4, 1e-3]

# Simulated metrics: for each (d,lambda) produce RMSE and Ops
data = []
for d in latent:
    for lam in lambda_vals:
        # ops scale with d and inversely with lambda (higher lambda -> sparser -> fewer ops)
        ops = int(1e6 * (d/32.0) * (1.0 / (1.0 + math.log10(lam*1e5+1))))
        rmse = 0.12 + 0.02*(32.0/d) + 0.01*(math.log10(lam*1e5+1))
        data.append((d, lam, ops, rmse))

with open(out, 'w') as f:
    f.write('% TikZ sweep figure (simulated)\n')
    f.write('\\begin{tikzpicture}[font=\\scriptsize]\n')
    f.write('  \\begin{axis}[width=\\columnwidth,height=3.2cm, xlabel={Ops (log10)}, ylabel={RMSE}, ymajorgrids, xmode=log]\n')
    # group by latent
    for d in latent:
        xs = []
        ys = []
        for (dd, lam, ops, rmse) in data:
            if dd==d:
                xs.append(ops)
                ys.append(rmse)
        f.write('    \\addplot+[mark=o] coordinates {')
        for x,y in zip(xs,ys):
            f.write('(%d,%0.3f) ' % (x,y))
        f.write('};\n')
    f.write('    \\legend{d=32,d=64,d=128}\n')
    f.write('  \\end{axis}\n')
    f.write('\\end{tikzpicture}\n')

print('Wrote', out)
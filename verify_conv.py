from pypdf import PdfReader
import subprocess, numpy as np
from PIL import Image
import os

for f in ["membranePotentialFull","membranePotentialDecay","membranePotentialIncrease"]:
    src = f"/home/ensismoebius/Repos/doutorado/documentation/00-thesis/monography/images/{f}.pdf"
    out = f"/tmp/opencode/{f}_out.pdf"
    r = PdfReader(out)
    fonts = set()
    for p in r.pages:
        fonts.update(list(p['/Resources'].get('/Font', {}).keys()))
    subprocess.run(["pdftoppm","-png","-r","150",out,f"/tmp/opencode/conv_{f}"], check=True)
    subprocess.run(["pdftoppm","-png","-r","150",src,f"/tmp/opencode/orig_{f}"], check=True)
    a = np.asarray(Image.open(f"/tmp/opencode/orig_{f}-1.png").convert('RGB')).astype(int)
    b = np.asarray(Image.open(f"/tmp/opencode/conv_{f}-1.png").convert('RGB')).astype(int)
    if a.shape != b.shape:
        a = a[:b.shape[0], :b.shape[1]]; b = b[:a.shape[0], :a.shape[1]]
    diff = np.abs(a-b).mean()
    print(f"{f}: fonts={list(fonts) or 'NONE'} size={r.pages[0].mediabox.width}x{r.pages[0].mediabox.height} mean_px_diff={diff:.4f}")

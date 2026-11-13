# notebooks/

Jupyter notebooks for prototyping and intuition: LFCC and MGDF voice features (`audio/`) and
the maths of a single neuron (`ml/`). They are scratch-pads. The experiments do not read
them, and no thesis or paper number comes from them.

The description of every notebook, what each needs, and its pitfalls (the audio notebooks
expect `voice01.wav` and `oi_lindo.wav` in `audio/`, which are **not in git**) are in
[`software/nn/.wiki/Notebooks.md`](../software/nn/.wiki/Notebooks.md).

```bash
python3 -m venv .venv && .venv/bin/pip install jupyter numpy scipy matplotlib soundfile scikit-learn ipywidgets
.venv/bin/jupyter notebook --notebook-dir notebooks/audio
```

"""Live microphone -> SNN vowel classifier (the lecture finale demo).

Unlike every other package in this app, this one drives a REAL trained
model from REAL, continuous, non-deterministic input (the microphone) --
see `demo.py`'s module docstring for how that breaks `core.demo.DemoModule`'s
usual "fully precomputed, no randomness" contract and what is done instead.
"""

from __future__ import annotations

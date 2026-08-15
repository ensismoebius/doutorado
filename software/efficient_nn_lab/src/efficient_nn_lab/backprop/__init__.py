"""Classic (non-quantized) forward/backward propagation demos.

Deliberately separate from bitnet/ and snn/: this is the plain-vanilla
mechanism (chain rule, gradient descent) that both BitNet's STE and the
SNN's surrogate gradient are variations *on top of*. Seeing it first, with
nothing exotic in the middle, makes it obvious exactly what those later
tricks are working around.
"""

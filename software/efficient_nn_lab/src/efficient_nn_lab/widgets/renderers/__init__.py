"""Per-demo renderer mixins for widgets/neuron_view.py's NeuronView.

One module per demo, each exporting a mixin with that demo's
`_render_*` method plus the constants and helpers only it uses.
NeuronView inherits all of them and keeps the `render()` dispatcher, so
the split changes where the code lives, not how it is called.
"""

from efficient_nn_lab.widgets.renderers._painting import PaintingMixin
from efficient_nn_lab.widgets.renderers.chain_layers import ChainLayersRendererMixin
from efficient_nn_lab.widgets.renderers.comparison import ComparisonRendererMixin
from efficient_nn_lab.widgets.renderers.matrix_algebra import MatrixAlgebraRendererMixin
from efficient_nn_lab.widgets.renderers.mlp import MlpNetworkRendererMixin
from efficient_nn_lab.widgets.renderers.pipelines import PipelineRenderersMixin

__all__ = [
    "PaintingMixin",
    "MlpNetworkRendererMixin",
    "MatrixAlgebraRendererMixin",
    "ChainLayersRendererMixin",
    "PipelineRenderersMixin",
    "ComparisonRendererMixin",
]

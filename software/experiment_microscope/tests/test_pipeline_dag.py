from experiment_microscope.views.pipeline_dag import _GRAPHS, PipelineDag, _NodeItem


def test_graphs_are_well_formed():
    for name, (nodes, edges) in _GRAPHS.items():
        keys = {n.key for n in nodes}
        assert len(keys) == len(nodes), f"{name}: duplicate node key"
        for a, b in edges:
            assert a in keys and b in keys, f"{name}: edge {a}->{b} references unknown node"


def test_dag_renders_and_active_nodes_carry_a_tab(qapp):
    dag = PipelineDag()
    dag.show_experiment("meeting01")
    node_items = [it for it in dag._scene.items() if isinstance(it, _NodeItem)]
    assert len(node_items) == len(_GRAPHS["meeting01"][0])
    active = [it.node for it in node_items if it.node.tab]
    assert active, "at least some meeting01 stages should be inspectable"


def test_node_click_emits_tab(qapp):
    dag = PipelineDag()
    seen = []
    dag.node_activated.connect(seen.append)
    dag.show_experiment("thesis")
    wavelet = next(
        it for it in dag._scene.items()
        if isinstance(it, _NodeItem) and it.node.key == "wavelet"
    )
    wavelet._on_click(wavelet.node.tab)
    assert seen == ["Wavelet Lab"]


def test_unknown_experiment_shows_placeholder(qapp):
    dag = PipelineDag()
    dag.show_experiment("paraconsistent_ga")
    assert not [it for it in dag._scene.items() if isinstance(it, _NodeItem)]

"""Specialize input dimensions in an ONNX copy without changing operations or weights."""
import hashlib
import json
from pathlib import Path

import onnx

from .runtime import read_model


def specialize_to_reference(source, reference, destination):
    source, reference, destination = Path(source), Path(reference), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('Shape specialization requires a new copy, never the original source')
    model = onnx.load(str(source))
    constants = {t.name for t in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in constants]
    graph = read_model(reference).subgraphs[0]
    if len(inputs) != 1 or len(graph.inputs) != 1:
        raise ValueError('Reference shape specialization requires an explicit single-input contract')
    tensor = inputs[0].type.tensor_type
    original = [d.dim_value if d.HasField('dim_value') else None for d in tensor.shape.dim]
    deployment = graph.tensors[int(graph.inputs[0])].shape.tolist()
    shape = [deployment[j] for j in [0, 3, 1, 2]] if len(original) == 4 else deployment
    if len(shape) != len(original) or any(d <= 0 for d in shape):
        raise ValueError('The reference does not provide a compatible fixed input rank')
    nodes = [node.SerializeToString() for node in model.graph.node]
    weights = [value.SerializeToString() for value in model.graph.initializer]
    for dim, size in zip(tensor.shape.dim, shape):
        dim.ClearField('dim_param')
        dim.dim_value = size
    # Old shape annotations can otherwise freeze the 416-pixel output grids.
    # Reshape controls and all other operator constants remain untouched.
    del model.graph.value_info[:]
    for output in model.graph.output:
        output.type.tensor_type.ClearField('shape')
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    assert nodes == [node.SerializeToString() for node in model.graph.node]
    assert weights == [value.SerializeToString() for value in model.graph.initializer]
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(destination))
    report = {'original_source': str(source.resolve()), 'specialized_copy': str(destination.resolve()),
              'reference': str(reference.resolve()), 'input': inputs[0].name,
              'original_input_shape': original, 'specialized_input_shape': shape,
              'operators_and_constants_unchanged': True,
              'original_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    destination.with_suffix('.specialization.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

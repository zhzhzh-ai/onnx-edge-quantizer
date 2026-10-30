"""Honor explicit terminal layout adapters in a supplied deployment reference."""
from copy import deepcopy

import numpy as np

from .runtime import modules, read_model, write_model, _normalized


def align_terminal_output_layout(path, reference, onnx_model, output_layouts):
    model, baseline = read_model(path), read_model(reference)
    _, s = modules()
    if len(model.subgraphs) != 1 or len(baseline.subgraphs) != 1:
        return []
    graph, ref = model.subgraphs[0], baseline.subgraphs[0]
    if len(graph.outputs) != len(ref.outputs) or len(graph.outputs) != len(onnx_model.graph.output):
        return []
    changes = []
    for source_id in list(map(int, graph.outputs)):
        source = graph.tensors[source_id]
        names = [value.name for value in onnx_model.graph.output
                 if _normalized(output_layouts.get(value.name, {}).get('tflite_name', value.name))
                 == _normalized(source.name)]
        if len(names) != 1:
            continue
        candidates = []
        for ref_id in map(int, ref.outputs):
            producer = next((op for op in ref.operators if ref_id in op.outputs), None)
            if producer is None:
                continue
            code = baseline.operatorCodes[producer.opcodeIndex]
            if max(code.builtinCode, code.deprecatedBuiltinCode) != s.BuiltinOperator.TRANSPOSE:
                continue
            if list(source.shape) == list(ref.tensors[int(producer.inputs[0])].shape):
                candidates.append(ref_id)
        # The explicit reference transpose supplies the axes; unique source
        # and target interface shapes establish multi-output correspondence.
        if len(candidates) != 1 or sum(list(graph.tensors[int(i)].shape) == list(source.shape)
                                       for i in graph.outputs) != 1:
            continue
        change = _align_one(model, baseline, source_id, candidates[0], names[0], output_layouts)
        if change:
            changes.append(change)
    if changes:
        write_model(model, path)
    return changes


def _align_one(model, baseline, source_id, ref_id, name, output_layouts):
    _, s = modules()
    graph, ref = model.subgraphs[0], baseline.subgraphs[0]
    producer = next((op for op in ref.operators if ref_id in op.outputs), None)
    if producer is None:
        return []
    code = baseline.operatorCodes[producer.opcodeIndex]
    if max(code.builtinCode, code.deprecatedBuiltinCode) != s.BuiltinOperator.TRANSPOSE:
        return []
    perm = ref.tensors[int(producer.inputs[1])]
    data = baseline.buffers[perm.buffer].data
    if perm.type != s.TensorType.INT32 or data is None:
        return []
    axes = np.frombuffer(bytes(data), '<i4').tolist()
    source = graph.tensors[source_id]
    if (axes != [0, 2, 3, 1] or len(source.shape) != 4
            or list(source.shape) != list(ref.tensors[int(producer.inputs[0])].shape)
            or list(source.shape) == list(ref.tensors[ref_id].shape)):
        return []
    target = deepcopy(source)
    source.name = source.name + b'/before_deployment_layout'
    target.shape = source.shape[axes].copy()
    target.shapeSignature = None
    graph.tensors.append(target)
    target_id = len(graph.tensors) - 1
    buffer = s.BufferT()
    buffer.data = np.frombuffer(np.array(axes, '<i4').tobytes(), np.uint8)
    model.buffers.append(buffer)
    perm_tensor = s.TensorT()
    perm_tensor.name, perm_tensor.shape, perm_tensor.type, perm_tensor.buffer = f'deployment_output_perm_{target_id}'.encode(), np.array([4], np.int32), s.TensorType.INT32, len(model.buffers) - 1
    graph.tensors.append(perm_tensor)
    entry = s.OperatorCodeT()
    entry.builtinCode = entry.deprecatedBuiltinCode = s.BuiltinOperator.TRANSPOSE
    entry.version = 1
    model.operatorCodes.append(entry)
    op = s.OperatorT()
    op.opcodeIndex = len(model.operatorCodes) - 1
    op.inputs, op.outputs = np.array([source_id, len(graph.tensors) - 1], np.int32), np.array([target_id], np.int32)
    op.builtinOptionsType, op.builtinOptions = s.BuiltinOptions.TransposeOptions, s.TransposeOptionsT()
    graph.operators.append(op)
    graph.outputs = np.array([target_id if int(i) == source_id else int(i) for i in graph.outputs], np.int32)
    for signature in model.signatureDefs or []:
        if signature.subgraphIndex == 0:
            for tensor in signature.outputs or []:
                if tensor.tensorIndex == source_id:
                    tensor.tensorIndex = target_id
    previous = output_layouts.get(name, {}).get('tflite_to_onnx_permutation', [0, 1, 2, 3])
    output_layouts[name] = {'tflite_name': target.name.decode(),
                            'tflite_to_onnx_permutation': [axes.index(i) for i in previous],
                            'source': 'explicit_reference_terminal_transpose',
                            'evidence': 'deployment_axis_contract'}
    return {'rewrite': 'reference_terminal_output_layout', 'output': name, 'permutation': axes}

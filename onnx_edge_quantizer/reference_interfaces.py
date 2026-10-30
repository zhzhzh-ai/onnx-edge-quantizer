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
        change = _absorb_terminal_transpose(model, baseline, source_id, names[0], output_layouts)
        if change:
            changes.append(change)
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


def _absorb_terminal_transpose(model, baseline, source_id, name, output_layouts):
    """Expose the pre-transpose output only when a unique reference requests it."""
    _, s = modules()
    graph, ref = model.subgraphs[0], baseline.subgraphs[0]
    producer = next((op for op in graph.operators if source_id in op.outputs), None)
    if producer is None:
        return None
    code = model.operatorCodes[producer.opcodeIndex]
    if (max(code.builtinCode, code.deprecatedBuiltinCode) != s.BuiltinOperator.TRANSPOSE
            or len(producer.inputs) != 2 or len(producer.outputs) != 1
            or any(source_id in op.inputs for op in graph.operators)):
        return None
    input_id = int(producer.inputs[0])
    source, inner = graph.tensors[source_id], graph.tensors[input_id]
    perm = graph.tensors[int(producer.inputs[1])]
    raw = model.buffers[perm.buffer].data
    if input_id in graph.outputs or perm.type != s.TensorType.INT32 or raw is None:
        return None
    axes = np.frombuffer(bytes(raw), '<i4').tolist()
    if (sorted(axes) != list(range(len(inner.shape)))
            or [int(inner.shape[i]) for i in axes] != list(source.shape)):
        return None
    targets = [int(i) for i in ref.outputs if list(ref.tensors[int(i)].shape) == list(inner.shape)]
    if (len(targets) != 1 or list(inner.shape) == list(source.shape)
            or any(list(ref.tensors[int(i)].shape) == list(source.shape) for i in ref.outputs)):
        return None
    # The reference asks for an interface already present in the candidate;
    # the explicit source transpose supplies the exact inverse for validation.
    inner.name = source.name
    graph.outputs = np.array([input_id if int(i) == source_id else int(i) for i in graph.outputs], np.int32)
    graph.operators = [op for op in graph.operators if op is not producer]
    for signature in model.signatureDefs or []:
        if signature.subgraphIndex == 0:
            for entry in signature.outputs or []:
                if entry.tensorIndex == source_id:
                    entry.tensorIndex = input_id
    previous = output_layouts.get(name, {}).get('tflite_to_onnx_permutation', list(range(len(axes))))
    output_layouts[name] = {'tflite_name': inner.name.decode(),
                            'tflite_to_onnx_permutation': [axes[i] for i in previous],
                            'source': 'explicit_candidate_terminal_transpose',
                            'evidence': 'reference_output_interface'}
    return {'rewrite': 'absorb_terminal_output_transpose', 'output': name, 'permutation': axes}


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
    if (axes not in ([0, 2, 3, 1], [0, 2, 1]) or len(source.shape) != len(axes)
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
    perm_tensor.name, perm_tensor.shape, perm_tensor.type, perm_tensor.buffer = f'deployment_output_perm_{target_id}'.encode(), np.array([len(axes)], np.int32), s.TensorType.INT32, len(model.buffers) - 1
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
    previous = output_layouts.get(name, {}).get('tflite_to_onnx_permutation', list(range(len(axes))))
    output_layouts[name] = {'tflite_name': target.name.decode(),
                            'tflite_to_onnx_permutation': [axes.index(i) for i in previous],
                            'source': 'explicit_reference_terminal_transpose',
                            'evidence': 'deployment_axis_contract'}
    return {'rewrite': 'reference_terminal_output_layout', 'output': name, 'permutation': axes}

"""Reproduce reference quantization boundaries only after core graph equality."""
from copy import deepcopy

import numpy as np

from .runtime import modules, read_model, write_model
from .structure import graph_contract
from .tflite_optimizations import compact_model


def _align_broadcast_shapes(model, reference):
    """Align singleton padding of constants; the caller proves full core equality."""
    _, s = modules()
    graph, ref = model.subgraphs[0], reference.subgraphs[0]
    if len(graph.operators) != len(ref.operators):
        return 0
    binary = {s.BuiltinOperator.ADD, s.BuiltinOperator.SUB, s.BuiltinOperator.MUL,
              s.BuiltinOperator.DIV, s.BuiltinOperator.MAXIMUM, s.BuiltinOperator.MINIMUM}
    dynamic = set(map(int, graph.inputs)) | {int(i) for op in graph.operators for i in op.outputs}
    ref_dynamic = set(map(int, ref.inputs)) | {int(i) for op in ref.operators for i in op.outputs}
    count = 0
    for op, target_op in zip(graph.operators, ref.operators):
        code, target_code = model.operatorCodes[op.opcodeIndex], reference.operatorCodes[target_op.opcodeIndex]
        kind = max(code.builtinCode, code.deprecatedBuiltinCode)
        if (kind not in binary or kind != max(target_code.builtinCode, target_code.deprecatedBuiltinCode)
                or len(op.inputs) != 2 or len(target_op.inputs) != 2):
            continue
        for slot in (0, 1):
            index, target_index = int(op.inputs[slot]), int(target_op.inputs[slot])
            if index < 0 or target_index < 0 or index in dynamic or target_index in ref_dynamic:
                continue
            tensor, target = graph.tensors[index], ref.tensors[target_index]
            shape = [] if tensor.shape is None else list(tensor.shape)
            wanted = [] if target.shape is None else list(target.shape)
            def trim_leading_ones(dims):
                dims = list(dims)
                while dims and dims[0] == 1:
                    dims.pop(0)
                return dims
            if shape == wanted or trim_leading_ones(shape) != trim_leading_ones(wanted):
                continue
            other, output = graph.tensors[int(op.inputs[1-slot])], graph.tensors[int(op.outputs[0])]
            try:
                if (tuple(np.broadcast_shapes(shape, other.shape)) != tuple(output.shape)
                        or tuple(np.broadcast_shapes(wanted, other.shape)) != tuple(output.shape)):
                    continue
            except ValueError:
                continue
            clone = deepcopy(tensor)
            clone.name = (clone.name or b'constant') + f'/reference_shape_{len(graph.tensors)}'.encode()
            clone.shape, clone.shapeSignature = np.array(wanted, np.int32), None
            graph.tensors.append(clone)
            op.inputs[slot] = len(graph.tensors) - 1
            count += 1
    return count


def align_quantize_boundaries(path, reference):
    model, baseline = read_model(path), read_model(reference)
    _, s = modules()
    if len(model.subgraphs) != 1 or len(baseline.subgraphs) != 1:
        return []

    def is_quantize(m, op):
        code = m.operatorCodes[op.opcodeIndex]
        # Float/index output conversions belong to the core graph. Only
        # integer-to-integer requantization boundaries can be reproduced as
        # identity encodings after proving the remaining graph is identical.
        graph = m.subgraphs[0]
        return (max(code.builtinCode, code.deprecatedBuiltinCode) == s.BuiltinOperator.QUANTIZE
                and all(graph.tensors[int(i)].type in (s.TensorType.INT8, s.TensorType.UINT8)
                        for i in [op.inputs[0], op.outputs[0]]))

    g, ref = model.subgraphs[0], baseline.subgraphs[0]
    if any(is_quantize(model, op) for op in g.operators):
        return []
    boundaries = [op for op in ref.operators if is_quantize(baseline, op)]
    if not boundaries:
        trial = deepcopy(model)
        compact_model(trial)
        compact_model(baseline)
        count = _align_broadcast_shapes(trial, baseline)
        if count and graph_contract(trial) == graph_contract(baseline):
            write_model(trial, path)
            return [{'rewrite': 'equivalent_reference_broadcast_shapes', 'count': count,
                     'core_graph_verified': True}]
        return []
    if any(int(op.outputs[0]) in ref.outputs for op in boundaries):
        return []
    if any(ref.tensors[int(i)].type not in (s.TensorType.INT8, s.TensorType.UINT8)
           for op in boundaries for i in [op.inputs[0], op.outputs[0]]):
        return []
    stripped = deepcopy(baseline)
    stripped_graph = stripped.subgraphs[0]
    aliases = {int(op.outputs[0]): int(op.inputs[0]) for op in stripped_graph.operators
               if is_quantize(stripped, op)}

    def resolve(i):
        while i in aliases:
            i = aliases[i]
        return i

    stripped_graph.operators = [op for op in stripped_graph.operators if not is_quantize(stripped, op)]
    for op in stripped_graph.operators:
        op.inputs = np.array([resolve(int(i)) for i in op.inputs], np.int32)
    trial = deepcopy(model)
    compact_model(trial)
    compact_model(stripped)
    shape_changes = _align_broadcast_shapes(trial, stripped)
    # Full operator, edge, shape and control-constant equality is required.
    if graph_contract(trial) != graph_contract(stripped):
        return []
    model = trial
    compact_model(model)
    compact_model(baseline)
    g, ref = model.subgraphs[0], baseline.subgraphs[0]
    candidate_ops = iter(g.operators)
    mapping = {int(r): int(c) for r, c in zip(ref.inputs, g.inputs)}
    rewritten, count = [], 0
    for reference_op in ref.operators:
        if is_quantize(baseline, reference_op):
            source = mapping[int(reference_op.inputs[0])]
            tensor = deepcopy(g.tensors[source])
            tensor.name = tensor.name + f'/reference_boundary_{count}'.encode()
            tensor.buffer = 0
            g.tensors.append(tensor)
            target = len(g.tensors) - 1
            code = s.OperatorCodeT()
            code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.QUANTIZE
            code.version = 2
            model.operatorCodes.append(code)
            op = s.OperatorT()
            op.opcodeIndex = len(model.operatorCodes) - 1
            op.inputs, op.outputs = np.array([source], np.int32), np.array([target], np.int32)
            rewritten.append(op)
            mapping[int(reference_op.outputs[0])] = target
            count += 1
            continue
        op = next(candidate_ops)
        op.inputs = np.array([mapping.get(int(r), int(c)) for r, c in zip(reference_op.inputs, op.inputs)], np.int32)
        mapping.update({int(r): int(c) for r, c in zip(reference_op.outputs, op.outputs)})
        rewritten.append(op)
    g.operators = rewritten
    compact_model(model)
    write_model(model, path)
    return [{'rewrite': 'reference_quantize_boundaries', 'count': count,
             'encoding': 'identity', 'core_graph_verified': True,
             'equivalent_broadcast_shapes': shape_changes}]

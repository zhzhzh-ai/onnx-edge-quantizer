"""Explicitly adopt a deployment reference's pooling boundary semantics."""
from copy import deepcopy

import numpy as np

from .runtime import modules, read_model, write_model
from .tflite_optimizations import compact_model


def adopt_reference_pool_padding(path, reference):
    """Opt-in: zero padding changes negative MaxPool / boundary AvgPool values.

    Only matching pool contracts and exact input/output geometries are used.
    This pass must never be described as an ONNX-equivalent optimization.
    """
    model, baseline = read_model(path), read_model(reference)
    _, s = modules()
    if len(model.subgraphs) != 1 or len(baseline.subgraphs) != 1:
        return []
    graph, ref = model.subgraphs[0], baseline.subgraphs[0]
    pools = {s.BuiltinOperator.MAX_POOL_2D, s.BuiltinOperator.AVERAGE_POOL_2D}

    def kind(m, op):
        code = m.operatorCodes[op.opcodeIndex]
        return max(code.builtinCode, code.deprecatedBuiltinCode)

    def options(op):
        p = op.builtinOptions
        return (p.strideH, p.strideW, p.filterHeight, p.filterWidth, p.fusedActivationFunction)

    def control(m, g, index):
        t = g.tensors[int(index)]
        raw = m.buffers[t.buffer].data
        return np.frombuffer(bytes(raw), '<i4') if t.type == s.TensorType.INT32 and raw is not None else None

    ref_producers = {int(i): op for op in ref.operators for i in op.outputs}
    templates = {}
    for pool in ref.operators:
        if kind(baseline, pool) not in pools or pool.builtinOptions.padding != s.Padding.VALID:
            continue
        pad = ref_producers.get(int(pool.inputs[0]))
        if pad is None or kind(baseline, pad) != s.BuiltinOperator.PAD:
            continue
        raw = control(baseline, ref, pad.inputs[1])
        shape = tuple(ref.tensors[int(pad.inputs[0])].shape)
        if raw is None or len(shape) != 4 or raw.size != 8:
            continue
        pads = raw.reshape(4, 2)
        if np.any(pads < 0) or np.any(pads[[0, 3]]):
            continue
        key = (kind(baseline, pool), shape, tuple(ref.tensors[int(pool.outputs[0])].shape), options(pool))
        templates.setdefault(key, {})[tuple(raw)] = pads

    producers = {int(i): op for op in graph.operators for i in op.outputs}
    users = {}
    for op in graph.operators:
        for i in op.inputs:
            users.setdefault(int(i), []).append(op)
    inserted, changes = {}, []
    for pool in graph.operators:
        pool_kind = kind(model, pool)
        if pool_kind not in pools:
            continue
        data = int(pool.inputs[0])
        parent = producers.get(data)
        explicit = parent is not None and kind(model, parent) == s.BuiltinOperator.PADV2
        source = int(parent.inputs[0]) if explicit else data
        tensor = graph.tensors[source]
        shape = tuple(tensor.shape) if tensor.shape is not None else ()
        target_shape = tuple(graph.tensors[int(pool.outputs[0])].shape)
        choices = templates.get((pool_kind, shape, target_shape, options(pool)), {})
        if len(choices) != 1 or tensor.type != s.TensorType.FLOAT32:
            continue
        pads = next(iter(choices.values()))
        if explicit:
            actual = control(model, graph, parent.inputs[1])
            fill = graph.tensors[int(parent.inputs[2])]
            raw_fill = model.buffers[fill.buffer].data
            if (data in graph.outputs or len(users[data]) != 1 or actual is None
                    or not np.array_equal(actual, pads.reshape(-1))
                    or fill.type != s.TensorType.FLOAT32 or raw_fill is None
                    or pool.builtinOptions.padding != s.Padding.VALID):
                continue
            value = np.frombuffer(bytes(raw_fill), '<f4')
            if value.size != 1 or np.isnan(value[0]):
                continue
            previous = ('-inf' if np.isneginf(value[0]) else '+inf' if np.isposinf(value[0])
                        else float(value[0]))
            code = s.OperatorCodeT()
            code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.PAD
            code.version = 1
            model.operatorCodes.append(code)
            parent.opcodeIndex = len(model.operatorCodes) - 1
            parent.inputs = parent.inputs[:2].copy()
            parent.builtinOptionsType, parent.builtinOptions = s.BuiltinOptions.PadOptions, s.PadOptionsT()
            changed = value[0] != 0
        else:
            if pool.builtinOptions.padding != s.Padding.SAME or len(shape) != 4:
                continue
            actual = np.zeros((4, 2), np.int32)
            p = pool.builtinOptions
            for axis, kernel, stride in [(1, p.filterHeight, p.strideH), (2, p.filterWidth, p.strideW)]:
                total = max((target_shape[axis] - 1) * stride + kernel - shape[axis], 0)
                actual[axis] = (total // 2, total - total // 2)
            if not np.array_equal(actual, pads):
                continue
            padded = deepcopy(tensor)
            padded.name = (tensor.name or b'pool_input') + f'/reference_zero_padding_{len(changes)}'.encode()
            padded.shape = np.array([d + int(p.sum()) for d, p in zip(shape, pads)], np.int32)
            padded.shapeSignature, padded.buffer = None, 0
            graph.tensors.append(padded)
            padded_id = len(graph.tensors) - 1
            raw = s.BufferT()
            raw.data = np.frombuffer(pads.astype('<i4').tobytes(), np.uint8)
            model.buffers.append(raw)
            control_tensor = s.TensorT()
            control_tensor.name = padded.name + b'/paddings'
            control_tensor.shape, control_tensor.type = np.array([4, 2], np.int32), s.TensorType.INT32
            control_tensor.buffer = len(model.buffers) - 1
            graph.tensors.append(control_tensor)
            code = s.OperatorCodeT()
            code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.PAD
            code.version = 1
            model.operatorCodes.append(code)
            pad = s.OperatorT()
            pad.opcodeIndex = len(model.operatorCodes) - 1
            pad.inputs, pad.outputs = np.array([source, len(graph.tensors) - 1], np.int32), np.array([padded_id], np.int32)
            pad.builtinOptionsType, pad.builtinOptions = s.BuiltinOptions.PadOptions, s.PadOptionsT()
            inserted[id(pool)] = pad
            pool.inputs[0] = padded_id
            pool.builtinOptions.padding = s.Padding.VALID
            previous, changed = 'implicit_same_pool_boundary', bool(np.any(pads))
        changes.append({'rewrite': 'adopt_reference_zero_pool_padding', 'input': tensor.name.decode(),
                        'previous_padding': previous, 'padding': pads.tolist(),
                        'semantics_changed': bool(changed), 'policy': 'baseline'})
    if changes:
        graph.operators = [part for op in graph.operators for part in ([inserted[id(op)], op] if id(op) in inserted else [op])]
        compact_model(model)
        write_model(model, path)
    return changes

"""Expand a channel permutation to the reference's proven shuffle contract."""
from copy import deepcopy

import numpy as np

from .runtime import modules, read_model, write_model
from .tflite_optimizations import compact_model


def align_channel_shuffles(path, reference):
    if reference is None:
        return []
    model, baseline = read_model(path), read_model(reference)
    _, s = modules()
    if len(model.subgraphs) != 1 or len(baseline.subgraphs) != 1:
        return []
    g, ref = model.subgraphs[0], baseline.subgraphs[0]

    def code(m, op):
        c = m.operatorCodes[op.opcodeIndex]
        return max(c.builtinCode, c.deprecatedBuiltinCode)

    def integers(m, graph, index):
        t = graph.tensors[int(index)]
        raw = m.buffers[t.buffer].data
        return np.frombuffer(bytes(raw), '<i4') if t.type == s.TensorType.INT32 and raw is not None else None

    producers = {int(i): op for op in ref.operators for i in op.outputs}
    patterns = {}
    for last in ref.operators:
        if code(baseline, last) != s.BuiltinOperator.RESHAPE:
            continue
        middle = producers.get(int(last.inputs[0]))
        if middle is None or code(baseline, middle) != s.BuiltinOperator.TRANSPOSE:
            continue
        first = producers.get(int(middle.inputs[0]))
        if first is None or code(baseline, first) != s.BuiltinOperator.RESHAPE:
            continue
        shape = tuple(ref.tensors[int(first.inputs[0])].shape)
        split = tuple(ref.tensors[int(first.outputs[0])].shape)
        axes = integers(baseline, ref, middle.inputs[1])
        if (len(shape) != 4 or len(split) != 5 or split[0] != shape[0]
                or split[3:] != shape[2:] or split[1] * split[2] != shape[1]
                or tuple(ref.tensors[int(last.outputs[0])].shape) != shape
                or axes is None or axes.tolist() != [0, 2, 1, 3, 4]):
            continue
        patterns.setdefault(shape, {})[split[1]] = (first, middle, last)
    replaced, count = [], 0
    dynamic = set(map(int, g.inputs)) | {int(i) for op in g.operators for i in op.outputs}
    for op in g.operators:
        if (code(model, op) != s.BuiltinOperator.GATHER or op.builtinOptions.axis != 1
                or op.builtinOptions.batchDims or int(op.inputs[1]) in dynamic):
            replaced.append(op)
            continue
        source, target = map(int, (op.inputs[0], op.outputs[0]))
        shape = tuple(g.tensors[source].shape)
        values = integers(model, g, op.inputs[1])
        matches = []
        for groups, pattern in patterns.get(shape, {}).items():
            wanted = np.arange(shape[1], dtype=np.int32).reshape(groups, -1).T.reshape(-1)
            if values is not None and np.array_equal(values, wanted):
                matches.append(pattern)
        if len(matches) != 1 or tuple(g.tensors[target].shape) != shape:
            replaced.append(op)
            continue
        current = source
        for step, template in enumerate(matches[0]):
            new_op = deepcopy(template)
            model.operatorCodes.append(deepcopy(baseline.operatorCodes[template.opcodeIndex]))
            new_op.opcodeIndex = len(model.operatorCodes) - 1
            control = deepcopy(ref.tensors[int(template.inputs[1])])
            model.buffers.append(deepcopy(baseline.buffers[control.buffer]))
            control.buffer = len(model.buffers) - 1
            control.name = f'channel_shuffle_{count}_{step}_control'.encode()
            g.tensors.append(control)
            new_op.inputs = np.array([current, len(g.tensors) - 1], np.int32)
            if step < 2:
                result = deepcopy(g.tensors[target])
                result.name = f'channel_shuffle_{count}_{step}'.encode()
                result.buffer = 0
                result.shape = ref.tensors[int(template.outputs[0])].shape.copy()
                result.shapeSignature = None
                g.tensors.append(result)
                current = len(g.tensors) - 1
            else:
                current = target
            new_op.outputs = np.array([current], np.int32)
            replaced.append(new_op)
        count += 1
    if count:
        g.operators = replaced
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'reference_channel_shuffle', 'count': count}] if count else []

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


def align_channel_shuffle_partitions(path, reference):
    """Expand full/partial Gather permutations using a proven reference chain.

    A complete integer index simulation proves that the chain only permutes
    channels and preserves every batch/spatial element. Partial gathers must
    equal contiguous slices of that exact permutation.
    """
    if reference is None:
        return []
    model, baseline = read_model(path), read_model(reference)
    _, s = modules()
    if len(model.subgraphs) != 1 or len(baseline.subgraphs) != 1:
        return []
    graph, ref = model.subgraphs[0], baseline.subgraphs[0]

    def kind(m, op):
        c = m.operatorCodes[op.opcodeIndex]
        return max(c.builtinCode, c.deprecatedBuiltinCode)

    def integers(m, g, index):
        t = g.tensors[int(index)]
        raw = m.buffers[t.buffer].data
        return np.frombuffer(bytes(raw), '<i4') if t.type == s.TensorType.INT32 and raw is not None else None

    producers = {int(i): op for op in ref.operators for i in op.outputs}
    transforms = {s.BuiltinOperator.RESHAPE, s.BuiltinOperator.TRANSPOSE}
    patterns = {}
    for last in ref.operators:
        if kind(baseline, last) not in transforms:
            continue
        output_shape = tuple(ref.tensors[int(last.outputs[0])].shape)
        if len(output_shape) != 4 or np.prod(output_shape) > 1_000_000:
            continue
        current, chain = last, []
        for _ in range(7):
            if current is None or kind(baseline, current) not in transforms or len(current.inputs) != 2:
                break
            chain.insert(0, current)
            source_shape = tuple(ref.tensors[int(current.inputs[0])].shape)
            if source_shape == output_shape and len(chain) >= 3:
                original = np.arange(np.prod(source_shape), dtype=np.int32).reshape(source_shape)
                actual = original
                try:
                    for step in chain:
                        control = integers(baseline, ref, step.inputs[1])
                        if control is None:
                            raise ValueError('dynamic transform control')
                        actual = (actual.reshape(tuple(control)) if kind(baseline, step) == s.BuiltinOperator.RESHAPE
                                  else actual.transpose(tuple(control)))
                        if actual.shape != tuple(ref.tensors[int(step.outputs[0])].shape):
                            raise ValueError('inconsistent transform shape')
                except (ValueError, IndexError):
                    break
                if np.array_equal(original, actual):
                    break
                for axis in (1, 3):
                    probe = [0] * 4
                    probe[axis] = slice(None)
                    values = actual[tuple(probe)].reshape(-1)
                    permutation = np.unravel_index(values, source_shape)[axis]
                    if (np.unique(permutation).size == source_shape[axis]
                            and np.array_equal(np.take(original, permutation, axis=axis), actual)):
                        key = (source_shape, axis, tuple(map(int, permutation)))
                        patterns.setdefault(key, chain)
                break
            current = producers.get(int(current.inputs[0]))

    dynamic = set(map(int, graph.inputs)) | {int(i) for op in graph.operators for i in op.outputs}
    groups = {}
    for op in graph.operators:
        if kind(model, op) == s.BuiltinOperator.GATHER and op.builtinOptions.batchDims == 0 and int(op.inputs[1]) not in dynamic:
            groups.setdefault(int(op.inputs[0]), []).append(op)
    removed, insertions, changes = set(), {}, []
    for source, gathers in groups.items():
        shape = tuple(graph.tensors[source].shape)
        options = []
        for (wanted_shape, axis, values), chain in patterns.items():
            if shape != wanted_shape:
                continue
            matched = []
            for op in gathers:
                indices = integers(model, graph, op.inputs[1])
                tensor = graph.tensors[int(op.inputs[1])]
                if (op.builtinOptions.axis % len(shape) != axis or indices is None
                        or tensor.shape is None or len(tensor.shape) != 1 or not indices.size):
                    continue
                starts = [i for i, value in enumerate(values) if value == indices[0]]
                if len(starts) == 1 and tuple(map(int, indices)) == values[starts[0]:starts[0] + len(indices)]:
                    target = list(shape)
                    target[axis] = len(indices)
                    if list(graph.tensors[int(op.outputs[0])].shape) == target:
                        matched.append((op, starts[0], len(indices)))
            if matched:
                options.append((axis, chain, matched))
        if len(options) != 1:
            continue
        axis, chain, matched = options[0]
        if len(matched) > 1 and any(count == shape[axis] for _, _, count in matched):
            continue
        ops, current = [], source

        def add_control(index, prefix):
            tensor = deepcopy(ref.tensors[int(index)])
            model.buffers.append(deepcopy(baseline.buffers[tensor.buffer]))
            tensor.buffer = len(model.buffers) - 1
            tensor.name = prefix
            tensor.quantization = None
            graph.tensors.append(tensor)
            return len(graph.tensors) - 1

        for n, template in enumerate(chain):
            prefix = f'shuffle_partition_{len(changes)}_{n}'.encode()
            op = deepcopy(template)
            model.operatorCodes.append(deepcopy(baseline.operatorCodes[op.opcodeIndex]))
            op.opcodeIndex = len(model.operatorCodes) - 1
            control_id = add_control(template.inputs[1], prefix + b'/control')
            result = deepcopy(graph.tensors[source])
            result.name, result.buffer = prefix, 0
            result.shape = ref.tensors[int(template.outputs[0])].shape.copy()
            result.shapeSignature = None
            graph.tensors.append(result)
            current, previous = len(graph.tensors) - 1, current
            op.inputs, op.outputs = np.array([previous, control_id], np.int32), np.array([current], np.int32)
            ops.append(op)
        for original, offset, count in matched:
            if offset == 0 and count == shape[axis]:
                # Reuse the public Gather output tensor to preserve all users.
                target = int(original.outputs[0])
                for op in ops:
                    op.outputs = np.array([target if int(i) == current else int(i) for i in op.outputs], np.int32)
                current = target
            else:
                begin, extent = np.zeros(4, np.int32), np.array(shape, np.int32)
                begin[axis], extent[axis] = offset, count
                controls = []
                for label, value in [('begin', begin), ('size', extent)]:
                    buffer = s.BufferT()
                    buffer.data = np.frombuffer(value.astype('<i4').tobytes(), np.uint8)
                    model.buffers.append(buffer)
                    tensor = s.TensorT()
                    tensor.name = f'shuffle_partition_{len(changes)}_{offset}/{label}'.encode()
                    tensor.type, tensor.shape, tensor.buffer = s.TensorType.INT32, np.array([4], np.int32), len(model.buffers) - 1
                    graph.tensors.append(tensor)
                    controls.append(len(graph.tensors) - 1)
                code = s.OperatorCodeT()
                code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.SLICE
                code.version = 1
                model.operatorCodes.append(code)
                op = s.OperatorT()
                op.opcodeIndex = len(model.operatorCodes) - 1
                op.inputs, op.outputs = np.array([current, *controls], np.int32), original.outputs.copy()
                ops.append(op)
            removed.add(id(original))
        insertions[id(matched[0][0])] = ops
        changes.append({'rewrite': 'reference_channel_shuffle_partitions', 'parts': len(matched),
                        'axis': axis, 'full_index_permutation_verified': True})
    if changes:
        graph.operators = [part for op in graph.operators for part in
                           (insertions.get(id(op), []) if id(op) in removed else [op])]
        compact_model(model)
        write_model(model, path)
    return changes

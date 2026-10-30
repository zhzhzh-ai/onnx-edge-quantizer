"""Reference-guided rewrites with explicit scalar and axis equivalence proofs."""
from copy import deepcopy

import numpy as np

from .runtime import modules, read_model, write_model
from .tflite_optimizations import compact_model


def align_scalar_transposes(path, reference):
    if reference is None:
        return []
    model, baseline = read_model(path), read_model(reference)
    _, s = modules()
    if len(model.subgraphs) != 1 or len(baseline.subgraphs) != 1:
        return []
    graph, ref = model.subgraphs[0], baseline.subgraphs[0]

    def kind(m, op):
        code = m.operatorCodes[op.opcodeIndex]
        return max(code.builtinCode, code.deprecatedBuiltinCode)

    binary = {s.BuiltinOperator.ADD, s.BuiltinOperator.SUB, s.BuiltinOperator.MUL,
              s.BuiltinOperator.DIV, s.BuiltinOperator.MAXIMUM, s.BuiltinOperator.MINIMUM}
    ref_producers = {int(i): op for op in ref.operators for i in op.outputs}
    targets = set()
    for transpose in ref.operators:
        if kind(baseline, transpose) != s.BuiltinOperator.TRANSPOSE:
            continue
        parent = ref_producers.get(int(transpose.inputs[0]))
        if parent is None or kind(baseline, parent) not in binary:
            continue
        perm = ref.tensors[int(transpose.inputs[1])]
        raw = baseline.buffers[perm.buffer].data
        if perm.type != s.TensorType.INT32 or raw is None:
            continue
        targets.add((kind(baseline, parent), tuple(ref.tensors[int(parent.outputs[0])].shape),
                     tuple(np.frombuffer(bytes(raw), '<i4'))))
    producers = {int(i): op for op in graph.operators for i in op.outputs}
    dynamic = set(producers) | set(map(int, graph.inputs))
    users = {}
    for op in graph.operators:
        for i in op.inputs:
            users.setdefault(int(i), []).append(op)
    count = 0
    for op in graph.operators:
        if kind(model, op) not in binary or len(op.inputs) != 2:
            continue
        for slot in (0, 1):
            middle, constant_id = int(op.inputs[slot]), int(op.inputs[1 - slot])
            transpose = producers.get(middle)
            if (transpose is None or kind(model, transpose) != s.BuiltinOperator.TRANSPOSE
                    or middle in graph.outputs or len(users.get(middle, [])) != 1
                    or constant_id in dynamic):
                continue
            c = graph.tensors[constant_id]
            raw = model.buffers[c.buffer].data
            if (c.type != s.TensorType.FLOAT32 or raw is None or len(raw) != 4
                    or np.prod(c.shape if c.shape is not None else []) != 1):
                continue
            source, output = int(transpose.inputs[0]), int(op.outputs[0])
            shape = graph.tensors[source].shape
            perm = graph.tensors[int(transpose.inputs[1])]
            perm_raw = model.buffers[perm.buffer].data
            if perm.type != s.TensorType.INT32 or perm_raw is None:
                continue
            axes = tuple(np.frombuffer(bytes(perm_raw), '<i4'))
            if ((kind(model, op), tuple(shape), axes) not in targets
                    or not np.array_equal(graph.tensors[middle].shape, graph.tensors[output].shape)
                    or len(c.shape if c.shape is not None else []) > len(shape)):
                continue
            graph.tensors[middle].shape = shape.copy()
            graph.tensors[middle].shapeSignature = None
            op.inputs[slot] = source
            op.outputs = np.array([middle], np.int32)
            transpose.inputs[0] = middle
            transpose.outputs = np.array([output], np.int32)
            count += 1
            break
    if count:
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'scalar_before_transpose', 'count': count}] if count else []


def align_reduction_indices(path, reference):
    if reference is None:
        return []
    model, baseline = read_model(path), read_model(reference)
    _, s = modules()
    graph, ref = model.subgraphs[0], baseline.subgraphs[0]
    reductions = {s.BuiltinOperator.REDUCE_MAX, s.BuiltinOperator.REDUCE_MIN,
                  s.BuiltinOperator.MEAN, s.BuiltinOperator.SUM,
                  s.BuiltinOperator.ARG_MAX, s.BuiltinOperator.ARG_MIN}

    def key(m, g, op):
        c = m.operatorCodes[op.opcodeIndex]
        kind = max(c.builtinCode, c.deprecatedBuiltinCode)
        if kind not in reductions or len(op.inputs) != 2:
            return None
        tensor = g.tensors[int(op.inputs[1])]
        raw = m.buffers[tensor.buffer].data
        shape = tuple(g.tensors[int(op.inputs[0])].shape)
        if not shape or tensor.type != s.TensorType.INT32 or raw is None:
            return None
        axes = np.frombuffer(bytes(raw), '<i4')
        if not axes.size or np.any(axes < -len(shape)) or np.any(axes >= len(shape)):
            return None
        return (kind, shape, tuple(sorted(int(i) % len(shape) for i in axes)),
                bool(getattr(op.builtinOptions, 'keepDims', False)))

    targets = {}
    for op in ref.operators:
        k = key(baseline, ref, op)
        if k is not None:
            targets.setdefault(k, []).append(op)
    changes, inserted = [], {}
    for op in graph.operators:
        k = key(model, graph, op)
        matches = targets.get(k, [])
        if len(matches) != 1:
            continue
        target = matches[0]
        tensor, control = graph.tensors[int(op.inputs[1])], ref.tensors[int(target.inputs[1])]
        raw, wanted = model.buffers[tensor.buffer].data, baseline.buffers[control.buffer].data
        if not np.array_equal(tensor.shape, control.shape) or bytes(raw) != bytes(wanted):
            clone = deepcopy(tensor)
            clone.name += b'/equivalent_reference_axes'
            clone.shape = control.shape.copy()
            clone.shapeSignature = None
            buffer = s.BufferT()
            buffer.data = np.asarray(wanted, np.uint8).copy()
            model.buffers.append(buffer)
            clone.buffer = len(model.buffers) - 1
            graph.tensors.append(clone)
            op.inputs[1] = len(graph.tensors) - 1
            changes.append({'rewrite': 'equivalent_reference_reduction_axis'})
        if (k[0] not in (s.BuiltinOperator.ARG_MAX, s.BuiltinOperator.ARG_MIN)
                or op.builtinOptions.outputType != s.TensorType.INT64
                or target.builtinOptions.outputType != s.TensorType.INT32
                or any(k[1][axis] > np.iinfo(np.int32).max for axis in k[2])):
            continue
        original = int(op.outputs[0])
        index = deepcopy(graph.tensors[original])
        index.name += b'/int32_indices'
        index.type = s.TensorType.INT32
        graph.tensors.append(index)
        index_id = len(graph.tensors) - 1
        op.outputs = np.array([index_id], np.int32)
        op.builtinOptions.outputType = s.TensorType.INT32
        code = s.OperatorCodeT()
        code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.CAST
        code.version = 1
        model.operatorCodes.append(code)
        cast = s.OperatorT()
        cast.opcodeIndex = len(model.operatorCodes) - 1
        cast.inputs, cast.outputs = np.array([index_id], np.int32), np.array([original], np.int32)
        cast.builtinOptionsType, cast.builtinOptions = s.BuiltinOptions.CastOptions, s.CastOptionsT()
        cast.builtinOptions.inDataType, cast.builtinOptions.outDataType = s.TensorType.INT32, s.TensorType.INT64
        inserted[id(op)] = cast
        changes.append({'rewrite': 'bounded_int32_argmax_with_int64_cast'})
    if changes:
        graph.operators = [part for op in graph.operators for part in ([op, inserted[id(op)]] if id(op) in inserted else [op])]
        compact_model(model)
        write_model(model, path)
    return changes


def align_flattened_concat(path, reference):
    """Use channel-major flattening branches while preserving the final array."""
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

    def axes(m, g, op):
        t = g.tensors[int(op.inputs[1])]
        raw = m.buffers[t.buffer].data
        return np.frombuffer(bytes(raw), '<i4').tolist() if t.type == s.TensorType.INT32 and raw is not None else None

    ref_producers = {int(i): op for op in ref.operators for i in op.outputs}
    targets = []
    for last in ref.operators:
        if kind(baseline, last) != s.BuiltinOperator.TRANSPOSE or axes(baseline, ref, last) != [0, 2, 1]:
            continue
        concat = ref_producers.get(int(last.inputs[0]))
        if concat is None or kind(baseline, concat) != s.BuiltinOperator.CONCATENATION or concat.builtinOptions.axis != 2:
            continue
        branches = []
        for index in concat.inputs:
            reshape = ref_producers.get(int(index))
            transpose = ref_producers.get(int(reshape.inputs[0])) if reshape is not None else None
            if (reshape is None or kind(baseline, reshape) != s.BuiltinOperator.RESHAPE
                    or transpose is None or kind(baseline, transpose) != s.BuiltinOperator.TRANSPOSE
                    or axes(baseline, ref, transpose) != [0, 3, 1, 2]):
                break
            branches.append(tuple(ref.tensors[int(transpose.inputs[0])].shape))
        if len(branches) == len(concat.inputs):
            targets.append(tuple(branches))
    producers = {int(i): op for op in graph.operators for i in op.outputs}
    users = {}
    for op in graph.operators:
        for i in op.inputs:
            users.setdefault(int(i), []).append(op)
    changes, additions = [], {}

    def control(value, name):
        value = np.array(value, '<i4')
        buf = s.BufferT()
        buf.data = np.frombuffer(value.tobytes(), np.uint8)
        model.buffers.append(buf)
        tensor = s.TensorT()
        tensor.name, tensor.type, tensor.shape = name, s.TensorType.INT32, np.array(value.shape, np.int32)
        tensor.buffer = len(model.buffers) - 1
        graph.tensors.append(tensor)
        return len(graph.tensors) - 1

    def transpose(source, target, permutation):
        code = s.OperatorCodeT()
        code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.TRANSPOSE
        code.version = 1
        model.operatorCodes.append(code)
        op = s.OperatorT()
        op.opcodeIndex = len(model.operatorCodes) - 1
        op.inputs = np.array([source, control(permutation, f'flatten_concat_perm_{target}'.encode())], np.int32)
        op.outputs = np.array([target], np.int32)
        op.builtinOptionsType, op.builtinOptions = s.BuiltinOptions.TransposeOptions, s.TransposeOptionsT()
        return op

    for concat in list(graph.operators):
        if kind(model, concat) != s.BuiltinOperator.CONCATENATION or concat.builtinOptions.axis != 1:
            continue
        branches, shapes = [], []
        for index in concat.inputs:
            reshape = producers.get(int(index))
            if (reshape is None or kind(model, reshape) != s.BuiltinOperator.RESHAPE
                    or int(index) in graph.outputs or len(users.get(int(index), [])) != 1):
                break
            shape = tuple(graph.tensors[int(reshape.inputs[0])].shape)
            if len(shape) != 4 or tuple(graph.tensors[int(index)].shape) != (shape[0], shape[1] * shape[2], shape[3]):
                break
            branches.append(reshape)
            shapes.append(shape)
        if len(branches) != len(concat.inputs) or tuple(shapes) not in targets:
            continue
        for reshape, shape in zip(branches, shapes):
            source = int(reshape.inputs[0])
            output = int(reshape.outputs[0])
            tensor = deepcopy(graph.tensors[source])
            tensor.name += b'/reference_flatten_layout'
            tensor.shape = np.array([shape[0], shape[3], shape[1], shape[2]], np.int32)
            tensor.shapeSignature, tensor.buffer = None, 0
            graph.tensors.append(tensor)
            target = len(graph.tensors) - 1
            additions[id(reshape)] = [transpose(source, target, [0, 3, 1, 2]), reshape]
            new_shape = [shape[0], shape[3], shape[1] * shape[2]]
            reshape.inputs = np.array([target, control(new_shape, tensor.name + b'/shape')], np.int32)
            graph.tensors[output].shape = np.array(new_shape, np.int32)
            graph.tensors[output].shapeSignature = None
            if reshape.builtinOptions is not None:
                reshape.builtinOptions.newShape = np.array(new_shape, np.int32)
        original = int(concat.outputs[0])
        tensor = deepcopy(graph.tensors[original])
        tensor.name += b'/channel_major'
        tensor.shape = tensor.shape[[0, 2, 1]].copy()
        tensor.shapeSignature, tensor.buffer = None, 0
        graph.tensors.append(tensor)
        target = len(graph.tensors) - 1
        concat.builtinOptions.axis = 2
        concat.outputs = np.array([target], np.int32)
        additions[id(concat)] = [concat, transpose(target, original, [0, 2, 1])]
        changes.append({'rewrite': 'reference_flattened_concat_layout', 'branches': len(branches)})
    if changes:
        graph.operators = [part for op in graph.operators for part in additions.get(id(op), [op])]
        compact_model(model)
        write_model(model, path)
    return changes


def align_pad_transposes(path, reference):
    if reference is None:
        return []
    model, baseline = read_model(path), read_model(reference)
    _, s = modules()
    graph, ref = model.subgraphs[0], baseline.subgraphs[0]

    def kind(m, op):
        code = m.operatorCodes[op.opcodeIndex]
        return max(code.builtinCode, code.deprecatedBuiltinCode)

    def integers(m, g, i):
        tensor = g.tensors[int(i)]
        raw = m.buffers[tensor.buffer].data
        return np.frombuffer(bytes(raw), '<i4') if tensor.type == s.TensorType.INT32 and raw is not None else None

    ref_producers = {int(i): op for op in ref.operators for i in op.outputs}
    targets = set()
    for pad in ref.operators:
        if kind(baseline, pad) != s.BuiltinOperator.PAD:
            continue
        transpose = ref_producers.get(int(pad.inputs[0]))
        if transpose is None or kind(baseline, transpose) != s.BuiltinOperator.TRANSPOSE:
            continue
        perm, pads = integers(baseline, ref, transpose.inputs[1]), integers(baseline, ref, pad.inputs[1])
        if perm is not None and pads is not None:
            targets.add((tuple(ref.tensors[int(transpose.inputs[0])].shape), tuple(perm), tuple(pads)))
    producers = {int(i): op for op in graph.operators for i in op.outputs}
    users = {}
    for op in graph.operators:
        for i in op.inputs:
            users.setdefault(int(i), []).append(op)
    changes = []
    for transpose in graph.operators:
        if kind(model, transpose) != s.BuiltinOperator.TRANSPOSE:
            continue
        middle = int(transpose.inputs[0])
        pad = producers.get(middle)
        if (pad is None or kind(model, pad) != s.BuiltinOperator.PAD
                or middle in graph.outputs or len(users.get(middle, [])) != 1):
            continue
        source, output = int(pad.inputs[0]), int(transpose.outputs[0])
        shape = graph.tensors[source].shape
        perm, pads = integers(model, graph, transpose.inputs[1]), integers(model, graph, pad.inputs[1])
        if (perm is None or pads is None or sorted(perm) != list(range(len(shape)))
                or pads.size != len(shape) * 2):
            continue
        reordered = pads.reshape(-1, 2)[perm].reshape(-1)
        if (tuple(shape), tuple(perm), tuple(reordered)) not in targets:
            continue
        control = deepcopy(graph.tensors[int(pad.inputs[1])])
        control.name += b'/before_transpose'
        raw = s.BufferT()
        raw.data = np.frombuffer(reordered.astype('<i4').tobytes(), np.uint8)
        model.buffers.append(raw)
        control.buffer = len(model.buffers) - 1
        graph.tensors.append(control)
        transpose.inputs[0] = source
        transpose.outputs = np.array([middle], np.int32)
        graph.tensors[middle].shape = shape[perm].copy()
        graph.tensors[middle].shapeSignature = None
        pad.inputs = np.array([middle, len(graph.tensors) - 1], np.int32)
        pad.outputs = np.array([output], np.int32)
        changes.append({'rewrite': 'transpose_before_padding', 'permutation': perm.tolist()})
    if changes:
        compact_model(model)
        write_model(model, path)
    return changes

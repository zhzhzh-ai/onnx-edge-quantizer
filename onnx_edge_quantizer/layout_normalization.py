"""Move layout changes through proven equivariant operators before calibration."""
from collections import Counter
from copy import deepcopy
import json

import numpy as np

from .runtime import modules, read_model, write_model
from .tflite_optimizations import compact_model


def normalize_channel_layouts(path, reference=None):
    report_path = path.with_name(path.stem.removesuffix('_float32') + '_tensor_correspondence_report.json')
    if not report_path.is_file():
        return []
    report = json.loads(report_path.read_text(encoding='utf-8'))
    reshape_nchw = {
        row['resolved_tflite_tensor_name'] for row in report.get('records', [])
        if row.get('onnx_op_type') == 'Reshape' and row.get('exists_in_final_model')
        and row.get('assume_channel_last_layout') is False
    }
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    g = model.subgraphs[0]
    shapes = {i: tuple(t.shape) if t.shape is not None else () for i, t in enumerate(g.tensors)}
    names = {v: k for k, v in vars(s.BuiltinOperator).items() if isinstance(v, int)}
    views, cache, operators, changes = {}, {}, [], Counter()
    original_ops = list(g.operators)
    produced = {int(i) for op in original_ops for i in op.outputs}
    nchw_to_nhwc = (0, 2, 3, 1)
    softmax_targets = []
    reshape_layouts = {}
    reduction_targets = set()
    if reference is not None:
        baseline = read_model(reference)
        ref = baseline.subgraphs[0]
        ref_producers = {int(i): op for op in ref.operators for i in op.outputs}
        for op in baseline.subgraphs[0].operators:
            c = baseline.operatorCodes[op.opcodeIndex]
            builtin = max(c.builtinCode, c.deprecatedBuiltinCode)
            if builtin in (s.BuiltinOperator.MEAN, s.BuiltinOperator.SUM,
                           s.BuiltinOperator.REDUCE_MAX, s.BuiltinOperator.REDUCE_MIN):
                t = ref.tensors[int(op.inputs[1])]
                raw = baseline.buffers[t.buffer].data
                if t.type == s.TensorType.INT32 and raw is not None:
                    shape = tuple(ref.tensors[int(op.inputs[0])].shape)
                    axes = tuple(sorted(int(i) % len(shape) for i in np.frombuffer(bytes(raw), '<i4')))
                    reduction_targets.add((builtin, shape, axes, bool(op.builtinOptions.keepDims)))
            if max(c.builtinCode, c.deprecatedBuiltinCode) == s.BuiltinOperator.SOFTMAX:
                softmax_targets.append(tuple(baseline.subgraphs[0].tensors[int(op.inputs[0])].shape))
            if max(c.builtinCode, c.deprecatedBuiltinCode) == s.BuiltinOperator.TRANSPOSE:
                parent = ref_producers.get(int(op.inputs[0]))
                if parent is None:
                    continue
                pc = baseline.operatorCodes[parent.opcodeIndex]
                tensor = ref.tensors[int(op.inputs[1])]
                raw = baseline.buffers[tensor.buffer].data
                if (max(pc.builtinCode, pc.deprecatedBuiltinCode) == s.BuiltinOperator.RESHAPE
                        and tensor.type == s.TensorType.INT32 and raw is not None):
                    shape = tuple(ref.tensors[int(op.inputs[0])].shape)
                    axes = tuple(map(int, np.frombuffer(bytes(raw), '<i4')))
                    if len(shape) in (4, 5) and sorted(axes) == list(range(len(shape))):
                        reshape_layouts.setdefault(shape, set()).add(axes)
    softmax_index = 0

    def identity(index):
        return tuple(range(len(shapes[int(index)])))

    def view(index):
        index = int(index)
        return views.get(index, (index, identity(index)))

    def constant(index, dtype='<i4'):
        t = g.tensors[int(index)]
        data = model.buffers[t.buffer].data
        expected_type = s.TensorType.FLOAT32 if dtype == '<f4' else s.TensorType.INT32
        if int(index) in produced or int(index) in g.inputs or data is None or t.type != expected_type:
            return None
        return np.frombuffer(bytes(data), dtype=dtype).reshape(shapes[int(index)])

    def add_constant(value, name, tensor_type=s.TensorType.INT32):
        value = np.asarray(value)
        buf = s.BufferT()
        buf.data = np.frombuffer(value.tobytes(), dtype=np.uint8)
        model.buffers.append(buf)
        t = s.TensorT()
        t.name, t.shape, t.type, t.buffer = name + f'/{len(g.tensors)}'.encode(), np.array(value.shape, np.int32), tensor_type, len(model.buffers) - 1
        g.tensors.append(t)
        return len(g.tensors) - 1

    transpose_code = next((i for i, c in enumerate(model.operatorCodes)
                           if max(c.builtinCode, c.deprecatedBuiltinCode) == s.BuiltinOperator.TRANSPOSE), None)

    def materialize(index, desired=None):
        nonlocal transpose_code
        index = int(index)
        desired = identity(index) if desired is None else tuple(desired)
        data, current = view(index)
        if current == desired:
            return data
        key = (index, desired)
        if key in cache:
            return cache[key]
        axes = tuple(current.index(axis) for axis in desired)
        target = deepcopy(g.tensors[index])
        target.name = target.name + f'/layout_{len(g.tensors)}'.encode()
        target.buffer = 0
        target.shape = np.array([shapes[index][axis] for axis in desired], np.int32)
        target.shapeSignature = None
        g.tensors.append(target)
        target_id = len(g.tensors) - 1
        if transpose_code is None:
            code = s.OperatorCodeT()
            code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.TRANSPOSE
            code.version = 1
            model.operatorCodes.append(code)
            transpose_code = len(model.operatorCodes) - 1
        op = s.OperatorT()
        op.opcodeIndex = transpose_code
        op.inputs = np.array([data, add_constant(np.array(axes, '<i4'), target.name + b'/perm')], np.int32)
        op.outputs = np.array([target_id], np.int32)
        op.builtinOptionsType, op.builtinOptions = s.BuiltinOptions.TransposeOptions, s.TransposeOptionsT()
        operators.append(op)
        cache[key] = target_id
        changes['inserted_transpose'] += 1
        return target_id

    def output_view(index, permutation):
        index = int(index)
        g.tensors[index].shape = np.array([shapes[index][axis] for axis in permutation], np.int32)
        g.tensors[index].shapeSignature = None
        views[index] = (index, permutation)

    unary = {'LOGISTIC', 'RELU', 'RELU6', 'LEAKY_RELU', 'ABS', 'NEG', 'TANH', 'EXP', 'LOG', 'SQRT', 'RSQRT'}
    binary = {'ADD', 'SUB', 'MUL', 'DIV', 'MAXIMUM', 'MINIMUM', 'SQUARED_DIFFERENCE'}
    for original in original_ops:
        op = deepcopy(original)
        entry = model.operatorCodes[op.opcodeIndex]
        code = names.get(max(entry.builtinCode, entry.deprecatedBuiltinCode), '')
        inputs = [int(i) for i in original.inputs]
        outputs = [int(i) for i in original.outputs]
        if code == 'TRANSPOSE':
            axes = constant(inputs[1])
            if axes is not None:
                data, permutation = view(inputs[0])
                axes = tuple(int(i) for i in axes.reshape(-1))
                if sorted(axes) == list(range(len(permutation))):
                    views[outputs[0]] = (data, tuple(axes.index(i) for i in permutation))
                    changes['absorbed_transpose'] += 1
                    continue
        permutation = None
        if code == 'SOFTMAX':
            old_shape = shapes[inputs[0]]
            if (softmax_index < len(softmax_targets) and len(old_shape) == 4
                    and softmax_targets[softmax_index] != old_shape
                    and tuple(old_shape[i] for i in (0, 2, 1, 3)) == softmax_targets[softmax_index]):
                # Rows of a last-axis softmax are independent: swapping its
                # two spatial axes commutes exactly with the operation.
                permutation = (0, 2, 1, 3)
                op.inputs[0] = materialize(inputs[0], permutation)
                changes['reference_softmax_row_layout'] += 1
            softmax_index += 1
        elif code in unary:
            permutation = view(inputs[0])[1]
            op.inputs[0] = materialize(inputs[0], permutation)
        elif code in binary:
            rank = len(shapes[outputs[0]])
            candidates = [view(i)[1] for i in inputs if len(shapes[i]) == rank]
            permutation = nchw_to_nhwc if rank == 4 and nchw_to_nhwc in candidates else (candidates[0] if candidates else tuple(range(rank)))
            transformed = []
            for index in inputs:
                if len(shapes[index]) == rank:
                    transformed.append(materialize(index, permutation))
                elif int(index) not in produced and int(index) not in g.inputs and g.tensors[index].type == s.TensorType.FLOAT32:
                    if permutation == tuple(range(rank)):
                        transformed.append(index)
                        continue
                    value = constant(index, '<f4')
                    if value is None:
                        break
                    padded = value.reshape((1,) * (rank - value.ndim) + value.shape)
                    value = padded.transpose(permutation)
                    # Keep broadcast constants compact; e.g. 1,C,1,1 -> C.
                    while value.ndim > 1 and value.shape[0] == 1:
                        value = value.reshape(value.shape[1:])
                    transformed.append(add_constant(value, g.tensors[index].name + b'/broadcast_layout', s.TensorType.FLOAT32))
                else:
                    break
            if len(transformed) == len(inputs):
                op.inputs = np.array(transformed, np.int32)
            else:
                permutation = None
        elif code in ('SPLIT', 'SPLIT_V'):
            data_slot, axis_slot = (1, 0) if code == 'SPLIT' else (0, 2)
            axis = constant(inputs[axis_slot])
            if axis is not None and axis.size == 1:
                permutation = view(inputs[data_slot])[1]
                op.inputs[data_slot] = materialize(inputs[data_slot], permutation)
                value = permutation.index(int(axis.item()) % len(permutation))
                op.inputs[axis_slot] = add_constant(np.array(value, '<i4').reshape(axis.shape),
                                                    g.tensors[inputs[axis_slot]].name + b'/layout')
        elif code == 'STRIDED_SLICE':
            p = op.builtinOptions
            candidate = view(inputs[0])[1]
            vectors = [constant(i) for i in inputs[1:]]
            if (len(vectors) == 3 and all(v is not None and v.size == len(candidate) for v in vectors)
                    and p.ellipsisMask == p.newAxisMask == p.shrinkAxisMask == 0
                    and not getattr(p, 'offset', False)):
                permutation = candidate
                op.inputs[0] = materialize(inputs[0], permutation)
                for slot, value in enumerate(vectors, 1):
                    op.inputs[slot] = add_constant(value.reshape(-1)[list(permutation)],
                                                    g.tensors[inputs[slot]].name + b'/strided_layout')
                for field in ('beginMask', 'endMask'):
                    mask = int(getattr(p, field))
                    setattr(p, field, sum(((mask >> axis) & 1) << j for j, axis in enumerate(permutation)))
        elif code in ('SLICE', 'PAD', 'PADV2'):
            permutation = view(inputs[0])[1]
            vectors = [constant(i) for i in inputs[1:3 if code == 'SLICE' else 2]]
            if all(v is not None for v in vectors):
                op.inputs[0] = materialize(inputs[0], permutation)
                for slot, value in enumerate(vectors, 1):
                    op.inputs[slot] = add_constant(value[list(permutation)], g.tensors[inputs[slot]].name + b'/layout')
                if code == 'PADV2':
                    op.inputs[2] = materialize(inputs[2])
            else:
                permutation = None
        elif code == 'CONCATENATION':
            rank = len(shapes[outputs[0]])
            candidates = [view(i)[1] for i in inputs]
            permutation = nchw_to_nhwc if rank == 4 and nchw_to_nhwc in candidates else candidates[0]
            op.inputs = np.array([materialize(i, permutation) for i in inputs], np.int32)
            axis = op.builtinOptions.axis % rank
            op.builtinOptions.axis = permutation.index(axis)
        elif code in ('MEAN', 'SUM', 'REDUCE_MAX', 'REDUCE_MIN'):
            axes = constant(inputs[1])
            if axes is not None:
                input_permutation = view(inputs[0])[1]
                rank = len(shapes[inputs[0]])
                reduced = {int(i) % rank for i in axes.reshape(-1)}
                builtin = max(entry.builtinCode, entry.deprecatedBuiltinCode)
                if (builtin, shapes[inputs[0]], tuple(sorted(reduced)), bool(op.builtinOptions.keepDims)) in reduction_targets:
                    input_permutation = identity(inputs[0])
                op.inputs[0] = materialize(inputs[0], input_permutation)
                op.inputs[1] = add_constant(np.array([input_permutation.index(i) for i in sorted(reduced)], '<i4'),
                                            g.tensors[inputs[1]].name + b'/reduction_layout')
                if op.builtinOptions.keepDims:
                    permutation = input_permutation
                else:
                    remaining = [i for i in range(rank) if i not in reduced]
                    permutation = tuple(remaining.index(i) for i in input_permutation if i not in reduced)
        if permutation is not None:
            for index in outputs:
                output_view(index, permutation)
            operators.append(op)
            if permutation != identity(outputs[0]):
                changes['layout_equivariant_op'] += 1
            continue
        # Operators with linear-order or layout-specific contracts retain
        # their original representation unless covered by an explicit rule.
        singleton_source = None
        if code == 'RESHAPE':
            source, p = view(inputs[0])
            old_shape = shapes[inputs[0]]
            if ([i for i in p if old_shape[i] != 1]
                    == [i for i in identity(inputs[0]) if old_shape[i] != 1]):
                singleton_source = source
        op.inputs = np.array([singleton_source if position == 0 and singleton_source is not None
                              else materialize(i) if i >= 0 else i
                              for position, i in enumerate(inputs)], np.int32)
        operators.append(op)
        for index in outputs:
            views[index] = (index, identity(index))
            cache[(index, identity(index))] = index
        if code == 'RESHAPE':
            index = outputs[0]
            name = g.tensors[index].name.decode()
            targets = reshape_layouts.get(shapes[index], set())
            desired = next(iter(targets)) if len(targets) == 1 else (
                nchw_to_nhwc if len(shapes[index]) == 4 and name in reshape_nchw else None)
            if desired is not None and index not in g.outputs:
                new_data = materialize(index, desired)
                views[index] = (new_data, desired)
                changes['reshape_layout_view'] += 1
    public_outputs = [(int(i), g.tensors[int(i)].name) for i in g.outputs]
    g.outputs = np.array([materialize(i) for i, _ in public_outputs], np.int32)
    for signature in model.signatureDefs or []:
        if signature.subgraphIndex == 0:
            for item in signature.outputs or []:
                item.tensorIndex = materialize(int(item.tensorIndex))
    for (old, name), new in zip(public_outputs, g.outputs):
        if old != int(new):
            g.tensors[old].name = (name or b'output') + f'/internal_{old}'.encode()
            g.tensors[int(new)].name = name
    g.operators = operators
    if changes:
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'normalize_channel_layout', 'kind': key, 'count': value}
            for key, value in sorted(changes.items())]


def lower_reference_splits(path, reference):
    """Express static partitions as the exact Slice contracts in a reference."""
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
    def control(m, graph, index):
        t = graph.tensors[int(index)]
        raw = m.buffers[t.buffer].data
        return np.frombuffer(bytes(raw), '<i4') if t.type == s.TensorType.INT32 and raw is not None else None
    patterns = Counter()
    for op in ref.operators:
        if code(baseline, op) != s.BuiltinOperator.SLICE:
            continue
        shape = tuple(ref.tensors[int(op.inputs[0])].shape)
        begin, size = (control(baseline, ref, i) for i in op.inputs[1:])
        if begin is not None and size is not None and len(begin) == len(shape):
            size = tuple(int(d-b) if n == -1 else int(n) for b, n, d in zip(begin, size, shape))
            patterns[(shape, tuple(map(int, begin)), size)] += 1
    replaced, count = [], 0
    dynamic = set(map(int, g.inputs)) | {int(i) for op in g.operators for i in op.outputs}
    for op in g.operators:
        kind = code(model, op)
        if kind not in (s.BuiltinOperator.SPLIT, s.BuiltinOperator.SPLIT_V):
            replaced.append(op)
            continue
        data, axis_id = (int(op.inputs[1]), int(op.inputs[0])) if kind == s.BuiltinOperator.SPLIT else (int(op.inputs[0]), int(op.inputs[2]))
        shape, axis = tuple(g.tensors[data].shape), control(model, g, axis_id)
        if axis is None or axis.size != 1 or axis_id in dynamic or not shape:
            replaced.append(op)
            continue
        axis = int(axis.item()) % len(shape)
        if kind == s.BuiltinOperator.SPLIT:
            sizes = [shape[axis] // len(op.outputs)] * len(op.outputs) if shape[axis] % len(op.outputs) == 0 else None
        else:
            values = control(model, g, op.inputs[1])
            sizes = list(map(int, values)) if values is not None and int(op.inputs[1]) not in dynamic else None
            if sizes is not None and sizes.count(-1) == 1:
                sizes[sizes.index(-1)] = shape[axis] - sum(n for n in sizes if n >= 0)
        parts, offset = [], 0
        if sizes is not None and len(sizes) == len(op.outputs) and all(n > 0 for n in sizes) and sum(sizes) == shape[axis]:
            for size in sizes:
                begin, extent = [0] * len(shape), list(shape)
                begin[axis], extent[axis] = offset, size
                parts.append((shape, tuple(begin), tuple(extent)))
                offset += size
        required = Counter(parts)
        if not parts or any(patterns[key] < number for key, number in required.items()):
            replaced.append(op)
            continue
        patterns.subtract(required)
        for output, (_, begin, size) in zip(op.outputs, parts):
            inputs = [data]
            for label, vector in [('begin', begin), ('size', size)]:
                buf = s.BufferT()
                buf.data = np.frombuffer(np.array(vector, '<i4').tobytes(), np.uint8)
                model.buffers.append(buf)
                t = s.TensorT()
                t.name, t.shape, t.type, t.buffer = f'reference_split_{output}_{label}'.encode(), np.array([len(shape)], np.int32), s.TensorType.INT32, len(model.buffers)-1
                g.tensors.append(t)
                inputs.append(len(g.tensors)-1)
            entry = s.OperatorCodeT()
            entry.builtinCode = entry.deprecatedBuiltinCode = s.BuiltinOperator.SLICE
            entry.version = 1
            model.operatorCodes.append(entry)
            part = s.OperatorT()
            part.opcodeIndex = len(model.operatorCodes)-1
            part.inputs, part.outputs = np.array(inputs, np.int32), np.array([output], np.int32)
            part.builtinOptionsType, part.builtinOptions = s.BuiltinOptions.SliceOptions, s.SliceOptionsT()
            replaced.append(part)
        count += 1
    if count:
        g.operators = replaced
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'reference_split_slices', 'count': count}] if count else []


def coalesce_slice_partitions(path, reference=None):
    """Replace a proven complete disjoint axis partition by one SPLIT_V."""
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    g = model.subgraphs[0]
    reference_groups = None
    if reference is not None:
        baseline = read_model(reference)
        ref = baseline.subgraphs[0]
        reference_groups = Counter()
        for op in ref.operators:
            entry = baseline.operatorCodes[op.opcodeIndex]
            if max(entry.builtinCode, entry.deprecatedBuiltinCode) != s.BuiltinOperator.SPLIT_V:
                continue
            controls = []
            for i in op.inputs[1:]:
                tensor = ref.tensors[int(i)]
                raw = baseline.buffers[tensor.buffer].data
                if tensor.type != s.TensorType.INT32 or raw is None:
                    break
                controls.append(np.frombuffer(bytes(raw), '<i4'))
            if len(controls) == 2:
                shape = tuple(ref.tensors[int(op.inputs[0])].shape)
                axis = int(controls[1].item()) % len(shape)
                reference_groups[(shape, axis, tuple(int(v) for v in controls[0]))] += 1
    groups = {}
    for op in g.operators:
        code = model.operatorCodes[op.opcodeIndex]
        if max(code.builtinCode, code.deprecatedBuiltinCode) != s.BuiltinOperator.SLICE:
            continue
        arrays = []
        for index in op.inputs[1:]:
            tensor = g.tensors[int(index)]
            data = model.buffers[tensor.buffer].data
            if tensor.type != s.TensorType.INT32 or data is None:
                break
            arrays.append(np.frombuffer(bytes(data), '<i4').copy())
        if len(arrays) != 2:
            continue
        source = int(op.inputs[0])
        shape = np.asarray(g.tensors[source].shape)
        begin, size = arrays
        size[size == -1] = shape[size == -1] - begin[size == -1]
        if len(begin) != len(shape) or np.any(size <= 0):
            continue
        changed = np.flatnonzero((begin != 0) | (size != shape))
        if len(changed) == 1:
            axis = int(changed[0])
            groups.setdefault((source, axis), []).append((int(begin[axis]), int(size[axis]), op))
    removed, replaced, count = set(), {}, 0
    for (source, axis), parts in groups.items():
        parts.sort(key=lambda item: item[0])
        offset = 0
        valid = len(parts) >= 2
        for begin, size, _ in parts:
            valid &= begin == offset
            offset += size
        if not valid or offset != g.tensors[source].shape[axis]:
            continue
        if reference_groups is not None:
            signature = (tuple(g.tensors[source].shape), axis, tuple(size for _, size, _ in parts))
            if not reference_groups[signature]:
                continue
            reference_groups[signature] -= 1

        def const(values, name):
            values = np.asarray(values, '<i4')
            buf = s.BufferT()
            buf.data = np.frombuffer(values.tobytes(), np.uint8)
            model.buffers.append(buf)
            tensor = s.TensorT()
            tensor.name, tensor.shape, tensor.type, tensor.buffer = name, np.array(values.shape, np.int32), s.TensorType.INT32, len(model.buffers) - 1
            g.tensors.append(tensor)
            return len(g.tensors) - 1

        code = s.OperatorCodeT()
        code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.SPLIT_V
        code.version = 1
        model.operatorCodes.append(code)
        split = s.OperatorT()
        split.opcodeIndex = len(model.operatorCodes) - 1
        prefix = f'partition_{source}_{axis}'.encode()
        split.inputs = np.array([source, const([size for _, size, _ in parts], prefix + b'/sizes'), const(axis, prefix + b'/axis')], np.int32)
        split.outputs = np.array([int(op.outputs[0]) for _, _, op in parts], np.int32)
        split.builtinOptionsType, split.builtinOptions = s.BuiltinOptions.SplitVOptions, s.SplitVOptionsT()
        split.builtinOptions.numSplits = len(parts)
        first = min((op for _, _, op in parts), key=lambda op: next(i for i, candidate in enumerate(g.operators) if candidate is op))
        replaced[id(first)] = split
        removed.update(id(op) for _, _, op in parts)
        count += 1
    if count:
        g.operators = [replaced[id(op)] if id(op) in replaced else op for op in g.operators
                       if id(op) not in removed or id(op) in replaced]
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'slice_partitions_to_split_v', 'count': count}] if count else []


def lower_nearest_resize_to_gathers(path):
    """Exact nearest-neighbor sampling with static NHWC dimensions."""
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    g = model.subgraphs[0]
    rewritten, count = [], 0
    for op in g.operators:
        code = model.operatorCodes[op.opcodeIndex]
        if max(code.builtinCode, code.deprecatedBuiltinCode) != s.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
            rewritten.append(op)
            continue
        data, size = (g.tensors[int(i)] for i in op.inputs)
        raw = model.buffers[size.buffer].data
        if data.shape is None or len(data.shape) != 4 or size.type != s.TensorType.INT32 or raw is None:
            rewritten.append(op)
            continue
        target_hw = np.frombuffer(bytes(raw), '<i4')
        if len(target_hw) != 2 or np.any(target_hw <= 0) or np.any(data.shape <= 0):
            rewritten.append(op)
            continue
        current, shape = int(op.inputs[0]), list(data.shape)
        for axis, target_dim in zip((1, 2), target_hw):
            source_dim = int(shape[axis])
            target_dim = int(target_dim)
            options = op.builtinOptions
            scale = ((source_dim - 1) / (target_dim - 1) if options.alignCorners and target_dim > 1
                     else source_dim / target_dim)
            locations = (np.arange(target_dim, dtype=np.float32) + (0.5 if options.halfPixelCenters else 0)) * np.float32(scale)
            indices = np.floor(locations + (0.5 if options.alignCorners else 0)).astype('<i4')
            indices = np.clip(indices, 0, source_dim - 1).astype('<i4')
            buffer = s.BufferT()
            buffer.data = np.frombuffer(indices.tobytes(), np.uint8)
            model.buffers.append(buffer)
            tensor = s.TensorT()
            tensor.name, tensor.shape, tensor.type, tensor.buffer = f'resize_{count}_{axis}_indices'.encode(), np.array([target_dim], np.int32), s.TensorType.INT32, len(model.buffers) - 1
            g.tensors.append(tensor)
            indices_id = len(g.tensors) - 1
            shape[axis] = target_dim
            if axis == 2:
                output = int(op.outputs[0])
            else:
                tensor = deepcopy(data)
                tensor.name = tensor.name + f'/resize_height_{count}'.encode()
                tensor.shape, tensor.shapeSignature, tensor.buffer = np.array(shape, np.int32), None, 0
                g.tensors.append(tensor)
                output = len(g.tensors) - 1
            entry = s.OperatorCodeT()
            entry.builtinCode = entry.deprecatedBuiltinCode = s.BuiltinOperator.GATHER
            entry.version = 1
            model.operatorCodes.append(entry)
            gather = s.OperatorT()
            gather.opcodeIndex = len(model.operatorCodes) - 1
            gather.inputs, gather.outputs = np.array([current, indices_id], np.int32), np.array([output], np.int32)
            gather.builtinOptionsType, gather.builtinOptions = s.BuiltinOptions.GatherOptions, s.GatherOptionsT()
            gather.builtinOptions.axis = axis
            rewritten.append(gather)
            current = output
        count += 1
    if count:
        g.operators = rewritten
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'nearest_resize_to_gathers', 'count': count}] if count else []


def collapse_transpose_chains(path):
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    g = model.subgraphs[0]
    removed, count = set(), 0
    for _ in range(8):
        producers = {int(i): op for op in g.operators if id(op) not in removed for i in op.outputs}
        consumers = {}
        for op in g.operators:
            if id(op) not in removed:
                for i in op.inputs:
                    consumers.setdefault(int(i), []).append(op)
        changed = False
        for op in g.operators:
            if id(op) in removed:
                continue
            c = model.operatorCodes[op.opcodeIndex]
            if max(c.builtinCode, c.deprecatedBuiltinCode) != s.BuiltinOperator.TRANSPOSE:
                continue
            source = int(op.inputs[0])
            parent = producers.get(source)
            if parent is None or source in g.outputs or len(consumers.get(source, [])) != 1:
                continue
            c = model.operatorCodes[parent.opcodeIndex]
            if max(c.builtinCode, c.deprecatedBuiltinCode) != s.BuiltinOperator.TRANSPOSE:
                continue
            axes = []
            for transform in (parent, op):
                tensor = g.tensors[int(transform.inputs[1])]
                raw = model.buffers[tensor.buffer].data
                if tensor.type != s.TensorType.INT32 or raw is None:
                    break
                axes.append(np.frombuffer(bytes(raw), '<i4'))
            if len(axes) != 2 or len(axes[0]) != len(axes[1]):
                continue
            combined = axes[0][axes[1]].astype('<i4')
            tensor = deepcopy(g.tensors[int(op.inputs[1])])
            tensor.name = tensor.name + f'/composed_{count}'.encode()
            buf = s.BufferT()
            buf.data = np.frombuffer(combined.tobytes(), np.uint8)
            model.buffers.append(buf)
            tensor.buffer = len(model.buffers) - 1
            g.tensors.append(tensor)
            op.inputs = np.array([parent.inputs[0], len(g.tensors) - 1], np.int32)
            removed.add(id(parent))
            count += 1
            changed = True
        if not changed:
            break
    if count:
        g.operators = [op for op in g.operators if id(op) not in removed]
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'compose_transposes', 'count': count}] if count else []


def normalize_unit_strided_slices(path):
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    g = model.subgraphs[0]
    count = 0
    for op in g.operators:
        entry = model.operatorCodes[op.opcodeIndex]
        if max(entry.builtinCode, entry.deprecatedBuiltinCode) != s.BuiltinOperator.STRIDED_SLICE:
            continue
        options = op.builtinOptions
        if options.ellipsisMask or options.newAxisMask or options.shrinkAxisMask or getattr(options, 'offset', False):
            continue
        shape = g.tensors[int(op.inputs[0])].shape
        arrays = []
        for index in op.inputs[1:]:
            tensor = g.tensors[int(index)]
            raw = model.buffers[tensor.buffer].data
            if tensor.type != s.TensorType.INT32 or raw is None:
                break
            arrays.append(np.frombuffer(bytes(raw), '<i4'))
        if len(arrays) != 3 or shape is None or len(arrays[0]) != len(shape) or np.any(arrays[2] != 1):
            continue
        begin, size = [], []
        for axis, (dimension, start, stop) in enumerate(zip(shape, arrays[0], arrays[1])):
            start = None if options.beginMask & (1 << axis) else int(start)
            stop = None if options.endMask & (1 << axis) else int(stop)
            start, stop, _ = slice(start, stop, 1).indices(int(dimension))
            begin.append(start)
            size.append(max(0, stop - start))
        if any(i <= 0 for i in size):
            continue
        controls = []
        for name, values in [('begin', begin), ('size', size)]:
            array = np.array(values, '<i4')
            buffer = s.BufferT()
            buffer.data = np.frombuffer(array.tobytes(), np.uint8)
            model.buffers.append(buffer)
            tensor = s.TensorT()
            tensor.name, tensor.shape, tensor.type, tensor.buffer = f'unit_slice_{count}_{name}'.encode(), np.array([len(array)], np.int32), s.TensorType.INT32, len(model.buffers) - 1
            g.tensors.append(tensor)
            controls.append(len(g.tensors) - 1)
        code = s.OperatorCodeT()
        code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.SLICE
        code.version = 1
        model.operatorCodes.append(code)
        op.opcodeIndex = len(model.operatorCodes) - 1
        op.inputs = np.array([op.inputs[0], *controls], np.int32)
        op.builtinOptionsType, op.builtinOptions = s.BuiltinOptions.SliceOptions, s.SliceOptionsT()
        count += 1
    if count:
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'unit_strided_slice_to_slice', 'count': count}] if count else []

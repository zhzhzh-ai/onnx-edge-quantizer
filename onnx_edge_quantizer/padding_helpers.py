"""Remove static zero-batch scaffolding emitted for dynamic ONNX padding."""
import numpy as np

from .runtime import modules, read_model, write_model
from .tflite_optimizations import compact_model


def simplify_padding_helpers(path, reference=None):
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    g = model.subgraphs[0]
    changes = []

    def code(m, op):
        c = m.operatorCodes[op.opcodeIndex]
        return max(c.builtinCode, c.deprecatedBuiltinCode)

    def constant(index, dtype):
        t = g.tensors[int(index)]
        raw = model.buffers[t.buffer].data
        expected_type = s.TensorType.FLOAT32 if dtype == '<f4' else s.TensorType.INT32
        if int(index) in produced or int(index) in g.inputs or raw is None or t.type != expected_type:
            return None
        return np.frombuffer(bytes(raw), dtype)

    same_convs = set()
    if reference is not None:
        baseline = read_model(reference)
        ref = baseline.subgraphs[0]
        for op in ref.operators:
            if (code(baseline, op) in (s.BuiltinOperator.CONV_2D, s.BuiltinOperator.DEPTHWISE_CONV_2D)
                    and op.builtinOptions.padding == s.Padding.SAME):
                same_convs.add((code(baseline, op), tuple(ref.tensors[int(op.inputs[0])].shape),
                                tuple(ref.tensors[int(op.inputs[1])].shape),
                                tuple(ref.tensors[int(op.outputs[0])].shape)))
    for _ in range(3):
        produced = {int(i) for op in g.operators for i in op.outputs}
        producers = {int(i): op for op in g.operators for i in op.outputs}
        users = {}
        for op in g.operators:
            for i in op.inputs:
                users.setdefault(int(i), []).append(op)
        aliases, removed = {}, set()
        for op in g.operators:
            kind, output = code(model, op), int(op.outputs[0])
            if kind == s.BuiltinOperator.ADD and output not in g.outputs and op.builtinOptions.fusedActivationFunction == 0:
                for slot in (0, 1):
                    index, other = int(op.inputs[slot]), int(op.inputs[1-slot])
                    data = constant(index, '<f4') if g.tensors[index].type == s.TensorType.FLOAT32 else None
                    if (data is not None and data.size and np.all(data == 0)
                            and np.array_equal(g.tensors[other].shape, g.tensors[output].shape)):
                        aliases[output] = other
                        removed.add(id(op))
                        changes.append({'rewrite': 'remove_zero_padding_add'})
                        break
            if kind == s.BuiltinOperator.SLICE and output not in g.outputs:
                parent = producers.get(int(op.inputs[0]))
                if (parent is not None and code(model, parent) == s.BuiltinOperator.CONCATENATION
                        and parent.builtinOptions.fusedActivationFunction == 0
                        and int(parent.outputs[0]) not in g.outputs
                        and len(users[int(parent.outputs[0])]) == 1):
                    begin, size = constant(op.inputs[1], '<i4'), constant(op.inputs[2], '<i4')
                    source = int(parent.inputs[0])
                    shape = g.tensors[source].shape
                    if (begin is not None and size is not None and np.all(begin == 0)
                            and np.array_equal(size, shape) and np.array_equal(g.tensors[output].shape, shape)):
                        aliases[output] = source
                        removed.update((id(op), id(parent)))
                        changes.append({'rewrite': 'remove_concat_prefix_slice'})
            if kind in (s.BuiltinOperator.CONV_2D, s.BuiltinOperator.DEPTHWISE_CONV_2D) and op.builtinOptions.padding == s.Padding.VALID:
                pad = producers.get(int(op.inputs[0]))
                if (pad is None or code(model, pad) != s.BuiltinOperator.PAD
                        or int(pad.outputs[0]) in g.outputs or len(users[int(pad.outputs[0])]) != 1):
                    continue
                source = int(pad.inputs[0])
                shape, weight, result = (g.tensors[i].shape for i in (source, int(op.inputs[1]), output))
                if (kind, tuple(shape), tuple(weight), tuple(result)) not in same_convs:
                    continue
                raw = constant(pad.inputs[1], '<i4')
                if raw is None or raw.size != 8:
                    continue
                pads = raw.reshape(4, 2)
                wanted = np.zeros((4, 2), np.int32)
                valid = True
                for axis, kernel, stride, dilation in (
                        (1, weight[1], op.builtinOptions.strideH, op.builtinOptions.dilationHFactor),
                        (2, weight[2], op.builtinOptions.strideW, op.builtinOptions.dilationWFactor)):
                    total = max((int(result[axis])-1)*stride + (int(kernel)-1)*dilation + 1-int(shape[axis]), 0)
                    wanted[axis] = (total//2, total-total//2)
                    valid &= int(result[axis]) == (int(shape[axis])+stride-1)//stride
                if valid and np.array_equal(pads, wanted):
                    op.inputs[0] = source
                    op.builtinOptions.padding = s.Padding.SAME
                    removed.add(id(pad))
                    changes.append({'rewrite': 'reference_same_conv_padding'})
        if not removed:
            break
        def resolve(i):
            while i in aliases:
                i = aliases[i]
            return i
        g.operators = [op for op in g.operators if id(op) not in removed]
        for op in g.operators:
            op.inputs = np.array([resolve(int(i)) for i in op.inputs], np.int32)
    if changes:
        compact_model(model)
        write_model(model, path)
    return changes


def deduplicate_float_padding(path):
    """Share identical pure FP32 padding of the same source tensor."""
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    g = model.subgraphs[0]
    dynamic = set(map(int, g.inputs)) | {int(i) for op in g.operators for i in op.outputs}
    seen, aliases, removed = {}, {}, set()
    for op in g.operators:
        code = model.operatorCodes[op.opcodeIndex]
        kind = max(code.builtinCode, code.deprecatedBuiltinCode)
        if (kind not in (s.BuiltinOperator.PAD, s.BuiltinOperator.PADV2)
                or int(op.outputs[0]) in g.outputs
                or g.tensors[int(op.inputs[0])].type != s.TensorType.FLOAT32):
            continue
        controls = []
        for i in op.inputs[1:]:
            t = g.tensors[int(i)]
            raw = model.buffers[t.buffer].data
            if int(i) in dynamic or raw is None:
                break
            controls.append((int(t.type), tuple(t.shape), bytes(raw)))
        if len(controls) != len(op.inputs) - 1:
            continue
        key = (kind, int(op.inputs[0]), tuple(controls), tuple(g.tensors[int(op.outputs[0])].shape))
        if key in seen:
            aliases[int(op.outputs[0])] = seen[key]
            removed.add(id(op))
        else:
            seen[key] = int(op.outputs[0])
    if not removed:
        return []
    for op in g.operators:
        op.inputs = np.array([aliases.get(int(i), int(i)) for i in op.inputs], np.int32)
    g.operators = [op for op in g.operators if id(op) not in removed]
    compact_model(model)
    write_model(model, path)
    return [{'rewrite': 'deduplicate_float_padding', 'count': len(removed)}]


def deduplicate_static_slices(path):
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    graph = model.subgraphs[0]
    dynamic = set(map(int, graph.inputs)) | {int(i) for op in graph.operators for i in op.outputs}
    seen, aliases, removed = {}, {}, set()
    for op in graph.operators:
        code = model.operatorCodes[op.opcodeIndex]
        kind = max(code.builtinCode, code.deprecatedBuiltinCode)
        if kind not in (s.BuiltinOperator.SLICE, s.BuiltinOperator.STRIDED_SLICE) or int(op.outputs[0]) in graph.outputs:
            continue
        controls = []
        for i in op.inputs[1:]:
            t = graph.tensors[int(i)]
            data = model.buffers[t.buffer].data
            if int(i) in dynamic or data is None:
                break
            controls.append((int(t.type), tuple(t.shape), bytes(data)))
        if len(controls) != len(op.inputs) - 1:
            continue
        p = op.builtinOptions
        options = tuple(int(getattr(p, name, 0)) for name in
                        ('beginMask', 'endMask', 'ellipsisMask', 'newAxisMask', 'shrinkAxisMask', 'offset'))
        source = int(op.inputs[0])
        while source in aliases:
            source = aliases[source]
        key = (kind, source, tuple(controls), options, tuple(graph.tensors[int(op.outputs[0])].shape))
        if key in seen:
            aliases[int(op.outputs[0])] = seen[key]
            removed.add(id(op))
        else:
            seen[key] = int(op.outputs[0])
    if removed:
        def resolve(i):
            while i in aliases:
                i = aliases[i]
            return i
        for op in graph.operators:
            op.inputs = np.array([resolve(int(i)) for i in op.inputs], np.int32)
        graph.operators = [op for op in graph.operators if id(op) not in removed]
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'deduplicate_static_slices', 'count': len(removed)}] if removed else []
